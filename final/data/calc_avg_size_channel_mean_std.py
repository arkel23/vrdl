import os
import argparse
import glob

import numpy as np
from PIL import Image
from PIL import ImageStat


def search_images(args):
    # the tuple of file types
    types = ('*.jpg', '*.jpeg', '*.png')
    files_all = []
    for file_type in types:
        # files_all is the list of files
        path = os.path.join(args.path, '**', file_type)
        files_curr_type = glob.glob(path, recursive=True)
        files_all.extend(files_curr_type)

        print(file_type, len(files_curr_type))

    print(f'Total image files in folder {args.path}: {len(files_all)}')
    return files_all


def read_image(fp):
    img = Image.open(fp)
    width, height = img.size
    r, g, b = ImageStat.Stat(img).mean
    return width, height, r, g, b


def calc_avg_size_channel_mean_std(args):
    files_all = search_images(args)

    width_list = []
    height_list = []
    channel_mean_list = []

    for fp in files_all:
        width, height, r, g, b = read_image(fp)
        width_list.append(width)
        height_list.append(height)
        channel_mean_list.append([r, g, b])

    avg_width = sum(width_list) / len(width_list)
    avg_height = sum(height_list) / len(height_list)
    channel_mean_array = np.array(channel_mean_list)
    channel_mean = np.mean(channel_mean_array, axis=0)
    channel_std = np.std(channel_mean_array, axis=0)

    print(f'Average width: {avg_width}\tAverage height: {avg_height}')
    print(f'Channel mean: {channel_mean}\tChannel std: {channel_std}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='path to image folder')
    args = parser.parse_args()

    calc_avg_size_channel_mean_std(args)


main()
