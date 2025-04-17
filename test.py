import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.backends import cudnn
from model.Net import Net
from utils.dataset import test_dataset
from utils.eval import eval
import matplotlib.pyplot as plt


def test(model, test_loader, save_path):
    model.eval()
    with torch.no_grad():
        for i in range(test_loader.size):
            image, gt, name, _ = test_loader.load_data()
            gt = np.asarray(gt, np.float32)

            image = image.cuda()

            output = model(image)
            output = F.interpolate(output[-1], size=gt.shape, mode='bilinear', align_corners=False)
            output = output.sigmoid().data.cpu().numpy().squeeze()
            output = (output - output.min()) / (output.max() - output.min() + 1e-8)

            plt.imsave(os.path.join(save_path, name), output, cmap='gray')

def evaluator(model, test_root, save_path, dataset_name, train_size=448):
    test_data_loader = test_dataset(image_root=test_root + 'Imgs/',
                                    gt_root=test_root + 'GT/',
                                    testsize=train_size,
                                    )
    test(model, test_data_loader, save_path)
    eval_score = eval(
        gt_path=test_root,
        pred_path=save_path,
        dataset_name=dataset_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--test_size', type=int, default=448)
    parser.add_argument('--pth_path', type=str, default='',
                        help='path to load your model checkpoint')
    parser.add_argument('--test_path', type=str, default='../dataset/TestDataset/', help='path to test dataset')
    opt = parser.parse_args()

    txt_save_path = './result/'
    os.makedirs(txt_save_path, exist_ok=True)

    print('>>> configs:', opt)

    # set the device for training
    if opt.device == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif opt.device == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1')

    cudnn.benchmark = True

    model = Net().cuda()

    model.load_state_dict(torch.load(opt.pth_path), strict=False)
    model.eval()

    for data_name in ['CAMO', 'CHAMELEON', 'COD10K', 'NC4K']:
        map_save_path = txt_save_path + "{}/".format(data_name)
        os.makedirs(map_save_path, exist_ok=True)
        evaluator(
            model=model,
            test_root=opt.test_path + data_name + '/',
            save_path=map_save_path,
            train_size=448,
            dataset_name=data_name)
