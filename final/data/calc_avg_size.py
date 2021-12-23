import os
import argparse
import glob
from PIL import Image


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
    return width, height


def calc_avg_size(args):
    files_all = search_images(args)

    width_list = []
    height_list = []

    for fp in files_all:
        width, height = read_image(fp)
        width_list.append(width)
        height_list.append(height)

    avg_width = sum(width_list) / len(width_list)
    avg_height = sum(height_list) / len(height_list)

    print(f'Average width: {avg_width}\tAverage height: {avg_height}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='path to image folder')
    args = parser.parse_args()

    calc_avg_size(args)


main()
