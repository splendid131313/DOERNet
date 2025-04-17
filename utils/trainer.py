import numpy as np
import torch
import wandb
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable

from logger_utils import AverageMeter
from loss import PixLoss
from metrics import Fmeasure, Emeasure, Smeasure


class Trainer:
    def __init__(self,
                 model,
                 train_loader,
                 test_loader,
                 optimizer,
                 scheduler,
                 cfg=None,
                 logger=None,
                 ):
        self.config = cfg
        self.logger = logger

        # init model
        self.model = model
        self.train_loader, self.test_loader = train_loader, test_loader
        self.optimizer = optimizer
        self.lr_scheduler = scheduler

        # Setting Losses
        self.criterion_gdt = nn.BCEWithLogitsLoss()
        self.loss = PixLoss()

        # others
        self.loss_log = AverageMeter()

        self.run = wandb.init(
            project="DOERNet",
            config=self.config,
        )

        self.table = wandb.Table(columns=["image", "pred", "target"])

    def _train_batch(self, batch):
        inputs = batch[0].to(self.config.device)
        gts = batch[1].to(self.config.device)

        preds = self.model(inputs)

        (outs_gdt_pred, outs_gdt_label), preds = preds

        for _idx, (_gdt_pred, _gdt_label) in enumerate(zip(outs_gdt_pred, outs_gdt_label)):
            _gdt_pred = nn.functional.interpolate(_gdt_pred, size=_gdt_label.shape[2:], mode='bilinear', align_corners=True).sigmoid()
            _gdt_label = _gdt_label.sigmoid()
            loss_gdt = self.criterion_gdt(_gdt_pred, _gdt_label) if _idx == 0 else self.criterion_gdt(_gdt_pred, _gdt_label) + loss_gdt
        self.loss_dict['loss_gdt'] = loss_gdt.item()

        # Loss
        loss_pix = self.loss(preds, torch.clamp(gts, 0, 1)) * 1.0
        self.loss_dict['loss_pix'] = loss_pix.item()

        # since there may be several losses for sal, the lambdas for them (lambdas_pix) are inside the loss.py
        loss = loss_pix
        loss = loss + loss_gdt * 1.0

        self.loss_log.update(loss.item(), inputs.size(0))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        inputs = inputs.cpu()
        preds = [tensor.cpu() for tensor in preds]
        gts = gts.cpu()

        self.table.add_data(wandb.Image(inputs[0].permute(1, 2, 0).detach().numpy() * 255),
                            [wandb.Image(o[0].permute(1, 2, 0).detach().numpy()) for o in preds],
                            wandb.Image(gts[0].permute(1, 2, 0).detach().numpy()))

        self.run.log(
            {'predictions_table': self.table,
             'loss': loss,
             'loss_gdt': loss_gdt,
             'loss_pix': loss_pix,
             'lr': self.optimizer.param_groups[0]['lr'],
             }, commit=True)

    def train_epoch(self, epoch):
        global logger_loss_idx
        self.model.train()
        self.loss_dict = {}

        for batch_idx, batch in enumerate(self.train_loader):
            self._train_batch(batch)
            # Logger
            if batch_idx % 100 == 0:
                info_progress = 'Epoch[{0}/{1}] Iter[{2}/{3}].'.format(epoch, self.config.epochs, batch_idx, len(self.train_loader))
                info_loss = 'Training Losses'
                for loss_name, loss_value in self.loss_dict.items():
                    info_loss += ', {}: {:.3f}'.format(loss_name, loss_value)
                self.logger.info(' '.join((info_progress, info_loss)))
        info_loss = '@==Final== Epoch[{0}/{1}]  Training Loss: {loss.avg:.3f}  '.format(epoch, self.config.epochs, loss=self.loss_log)
        self.logger.info(info_loss)

        self.lr_scheduler.step()

        return self.loss_log.avg

    def train(self):

        for epoch in range(1, self.config.epochs + 1):
            self.train_epoch(epoch)
            if epoch % 10 == 0:
                torch.save(self.model.state_dict(), self.config.save_path + 'Net_epoch_{}.pth'.format(epoch))

            # Val
            if epoch == 1 or epoch % 5 == 0 or (epoch >= self.config.epochs * 0.8):
                self.val(epoch)

        wandb.finish()

    def val(self, epoch):
        """
        validation function
        """
        global best_metric_dict, best_score, best_epoch
        FM = Fmeasure()
        SM = Smeasure()
        EM = Emeasure()
        metrics_dict = dict()

        self.model.eval()
        with torch.no_grad():
            for i in range(self.test_loader.size):
                image, gt, _, _ = self.test_loader.load_data()
                gt = np.asarray(gt, np.float32)
                image = image.cuda()

                res = self.model(image)

                res = F.upsample(res[-1], size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)

                FM.step(pred=res, gt=gt)
                SM.step(pred=res, gt=gt)
                EM.step(pred=res, gt=gt)

            metrics_dict.update(Sm=SM.get_results()['sm'])
            metrics_dict.update(mxFm=FM.get_results()['fm']['curve'].max().round(3))
            metrics_dict.update(mxEm=EM.get_results()['em']['curve'].max().round(3))

            cur_score = metrics_dict['Sm'] + metrics_dict['mxFm'] + metrics_dict['mxEm']

            if epoch == 1:
                best_score = cur_score
                best_epoch = epoch
                best_metric_dict = metrics_dict
                print('[Cur Epoch: {}] Metrics (mxFm={}, Sm={}, mxEm={})'.format(
                    epoch, metrics_dict['mxFm'], metrics_dict['Sm'], metrics_dict['mxEm']))
                self.logger.info('[Cur Epoch: {}] Metrics (mxFm={}, Sm={}, mxEm={})'.format(
                    epoch, metrics_dict['mxFm'], metrics_dict['Sm'], metrics_dict['mxEm']))
            else:
                if cur_score > best_score:
                    best_metric_dict = metrics_dict
                    best_score = cur_score
                    best_epoch = epoch
                    torch.save(self.model.state_dict(), self.config.save_path + 'Net_epoch_best.pth')
                    print('>>> save state_dict successfully! best epoch is {}.'.format(epoch))
                else:
                    print('>>> not find the best epoch -> continue training ...')
                print(
                    '[Cur Epoch: {}] Metrics (mxFm={}, Sm={}, mxEm={})\n[Best Epoch: {}] Metrics (mxFm={}, Sm={}, mxEm={})'.format(
                        epoch, metrics_dict['mxFm'], metrics_dict['Sm'], metrics_dict['mxEm'],
                        best_epoch, best_metric_dict['mxFm'], best_metric_dict['Sm'], best_metric_dict['mxEm']))
                self.logger.info(
                    '[Cur Epoch: {}] Metrics (mxFm={}, Sm={}, mxEm={})\n[Best Epoch:{}] Metrics (mxFm={}, Sm={}, mxEm={})'.format(
                        epoch, metrics_dict['mxFm'], metrics_dict['Sm'], metrics_dict['mxEm'],
                        best_epoch, best_metric_dict['mxFm'], best_metric_dict['Sm'], best_metric_dict['mxEm']))