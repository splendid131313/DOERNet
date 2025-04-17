import argparse

import torch
import wandb
from torch import optim, nn
from torch.backends import cudnn

from utils.dataset import *
from model.Net import Net
from utils.trainer import Trainer
from utils.logger_utils import Logger


if __name__ == "__main__":
    import sys

    # sys.path.append("G:\CV\Models\MyNet\MyNet")
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--train_root', type=str, default='../dataset/TrainDataset/')
    parser.add_argument('--val_root', type=str, default='../dataset/TestDataset/CAMO/')
    parser.add_argument('--img_size', type=int, default=448)
    parser.add_argument('--ckpt_dir', default='./logs', help='Temporary folder')
    parser.add_argument('--save_path', type=str, default='checkpoints')

    config = parser.parse_args()

    # set the device for training
    if config.device == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif config.device == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1')
    elif config.device == 2:
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        print('USE GPU 2')
    elif config.device == 3:
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        print('USE GPU 3')
    cudnn.benchmark = True

    # make dir for ckpt
    os.makedirs(config.ckpt_dir, exist_ok=True)

    # Init log file
    logger = Logger(os.path.join(config.ckpt_dir, "log.txt"))
    logger_loss_idx = 1
    # log model and optimizer params
    # logger.info("Model details:"); logger.info(model)
    logger.info("Other hyperparameters:");
    logger.info(config)
    print('batch size:', config.batch_size)

    print('load data...')
    train_loader = get_loader(image_root=config.train_root + 'Imgs/',
                              gt_root=config.train_root + 'GT/',
                              batch_size=config.batch_size,
                              train_size=config.img_size,
                              num_workers=config.num_workers)
    val_loader = test_dataset(image_root=config.val_root + 'Imgs/',
                              gt_root=config.val_root + 'GT/',
                              testsize=config.img_size)
    total_step = len(train_loader)

    model = Net().cuda()

    optimizer = optim.AdamW(params=model.parameters(), lr=config.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-5)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=config,
        img_size=config.img_size,
        logger=logger,
    )

    trainer.train()



