import os
import argparse
import glob
from math import gcd
from functools import reduce

from PIL import Image

MIN_SIZE = 144


def lcm(denominators):
    return reduce(lambda a, b: a*b // gcd(a, b), denominators)


def find_closest_divisible(n, m):
    q = int(n / m)
    n1 = m * q
    n2 = m * (q + 1)

    if abs(n - n1) < abs(n - n2):
        return n1
    return n2


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

    print('Total image files pre-filtering', len(files_all))
    return files_all


def read_image(fp):
    img = Image.open(fp)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    width, height = img.size

    return img, width, height


def resize_img_div_by(img, width, height, common_denominator):
    if width % common_denominator != 0:
        new_width = find_closest_divisible(width, m=common_denominator)
    else:
        new_width = width
    if height % common_denominator != 0:
        new_height = find_closest_divisible(height, m=common_denominator)
    else:
        new_height = height
    img_resized_divisible = img.resize((new_width, new_height), Image.BICUBIC)
    return img_resized_divisible, new_width, new_height


def save_img(img, root_path, folder_name, fn):
    new_fn = os.path.join(root_path, folder_name, fn)
    img.save(new_fn)


def rescale_image_by_factor(img, new_width, new_height, scale_factor):
    scaled_width = int(new_width / scale_factor)
    scaled_height = int(new_height / scale_factor)
    img_rescaled = img.resize((scaled_width, scaled_height), Image.BICUBIC)
    return img_rescaled


def resize_images(args):
    files_all = search_images(args)

    # common denominator to scale_factors (all image sizes must be div by this)
    common_denominator = lcm(args.scale_factors)

    root_path, _ = os.path.split(os.path.normpath(args.path_images))
    folder_HR = f'HW4_train_HR_divby{common_denominator}'
    os.makedirs(os.path.join(root_path, folder_HR), exist_ok=True)

    for scale_factor in args.scale_factors:
        folder_LR = 'HW4_train_LR_bicubic'
        folder_name = os.path.join(
            root_path, 'HW4_train_LR_bicubic', f'X{scale_factor}')
        os.makedirs(folder_name, exist_ok=True)

    j = 0
    for i, fp in enumerate(files_all):
        abs_path, fn = os.path.split(os.path.normpath(fp))
        img, width, height = read_image(fp)

        img_resized, new_width, new_height = resize_img_div_by(
            img, width, height, common_denominator)

        if new_width <= MIN_SIZE or new_height <= MIN_SIZE:
            continue

        j += 1
        save_img(img_resized, root_path, folder_HR, fn)

        for scale_factor in args.scale_factors:
            img_rescaled = rescale_image_by_factor(
                img, new_width, new_height, scale_factor)
            name = os.path.splitext(os.path.basename(os.path.normpath(fp)))[0]
            # new_fn = f'{name}x{scale_factor}.png'
            folder_name = os.path.join(folder_LR, f'X{scale_factor}')
            save_img(img_rescaled, root_path, folder_name, fn)

        if i % args.print_freq == 0:
            print(f'{i}/{len(files_all)}: {fp}')
    print(f'Total images post-filtering: {j} (Min size: {MIN_SIZE})')


def main():
    '''takes a folder with images and resizes by a factor'''

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_images', type=str, help='folder with images')
    parser.add_argument('--scale_factors', type=list, default=[2, 3],
                        help='rescale factor for images')
    parser.add_argument('--print_freq', type=int, default=50, help='printfreq')
    args = parser.parse_args()

    resize_images(args)


main()
