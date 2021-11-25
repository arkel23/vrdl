import os
import time
import argparse
from tqdm import tqdm

import cv2
import torch
from mmdet.apis import init_detector, inference_detector
from google_drive_downloader import GoogleDriveDownloader as gdd


TEST_IMAGE_NUMBER = 100
DEFAULT_CKPT = 'vfnet_r101_fpn_666x400_svhn_epoch3.pth'
DEFAULT_CFG = 'vfnet_r101_fpn_mdconv_c3-c5_mstrain_2x_svhn_666x400.py'


def download_ckpt():
    print('Downloading default checkpoint')
    gdd.download_file_from_google_drive(
        file_id='1XK7YfK1ImlhZXY62CO8omJVful-tGGAV',
        dest_path=os.path.join(
            'checkpoints', DEFAULT_CKPT),
        unzip=False)


def verify_download(args):
    if not os.path.isfile(args.ckpt):
        print('Using default config')
        args.config = os.path.join('configs', 'custom', DEFAULT_CFG)

        os.makedirs('checkpoints', exist_ok=True)
        args.ckpt = os.path.join('checkpoints', DEFAULT_CKPT)
        if not os.path.isfile(args.ckpt):
            download_ckpt()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=os.path.join(
            'configs', 'custom', DEFAULT_CFG),
        help='path to config file')
    parser.add_argument(
        '--ckpt', type=str, default=os.path.join(
            'checkpoints', DEFAULT_CKPT),
        help='path to checkpoint file')
    parser.add_argument('--data', type=str, default='data/test',
                        help='path to test data folder')
    args = parser.parse_args()

    verify_download(args)
    config_file = args.config
    checkpoint_file = args.ckpt
    print(args)
    data_listdir = os.listdir(args.data)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    test_img_list = []
    data_listdir.sort(key=lambda x: int(x[:-4]))

    for img_name in data_listdir[:TEST_IMAGE_NUMBER]:
        img_path = os.path.join(args.data, img_name)
        img = cv2.imread(img_path)
        test_img_list.append(img)

    model = init_detector(config_file, checkpoint_file, device=device)

    start_time = time.time()
    for img in tqdm(test_img_list):
        pred = inference_detector(model, img)

    end_time = time.time()
    avg_time = (end_time - start_time) / len(test_img_list)
    print("\nInference time per image: ", avg_time)


if __name__ == '__main__':
    main()
