import os
import glob
import argparse

import mmcv
import torch

from mmedit.apis import init_model, restoration_inference
from mmedit.core import tensor2img


def search_images(args):
    # the tuple of file types
    types = ('*.jpg', '*.jpeg', '*.png')
    files_all = []
    for file_type in types:
        # files_all is the list of files
        path = os.path.join(args.path_images, '**', file_type)
        files_curr_type = glob.glob(path, recursive=True)
        files_all.extend(files_curr_type)

        print(file_type, len(files_curr_type))

    print('Total image files: ', len(files_all))
    return files_all


def parse_args():
    parser = argparse.ArgumentParser(description='Restoration demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('path_images', help='path to folder with images')
    parser.add_argument('save_path', help='path to folder to save results')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    model = init_model(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))

    files_all = search_images(args)

    for fp in files_all:
        output = restoration_inference(model, fp)
        output = tensor2img(output)

        name = os.path.splitext(os.path.basename(os.path.normpath(fp)))[0]
        new_fn = f'{name}_pred.png'
        mmcv.imwrite(output, os.path.join(args.save_path, new_fn))
        print(args.save_path, new_fn)


if __name__ == '__main__':
    main()
