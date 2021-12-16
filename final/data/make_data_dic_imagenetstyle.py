import os
import argparse
import glob
import pandas as pd
import numpy as np
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

    print('Total image files pre-filtering', len(files_all))
    return files_all


def add_image_to_dics(idx, fp, classid_classname_dic, filename_classid_dic):
    # verify the image is RGB
    img = Image.open(fp)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    if img.mode == 'RGB':
        abs_path, filename = os.path.split(fp)
        _, class_name = os.path.split(abs_path)
        rel_path = os.path.join(class_name, filename)

        if class_name not in classid_classname_dic.values():
            idx += 1
            classid_classname_dic[idx] = class_name

        filename_classid_dic[rel_path] = idx

    return idx


def save_classid_classname(classid_classname_dic):
    df_classid_classname = pd.DataFrame.from_dict(
        classid_classname_dic, orient='index', columns=['class_name'])

    # change index
    idx_col = np.arange(0, len(df_classid_classname), 1)
    df_classid_classname['idx_col'] = idx_col
    df_classid_classname['class_id'] = df_classid_classname.index
    df_classid_classname.set_index('idx_col', inplace=True)

    print(df_classid_classname.head())
    fn = 'classid_classname' + '.csv'
    df_classid_classname.to_csv(fn, sep=',', header=True, index=False)


def save_filename_classid(args, filename_classid_dic):
    # save dataframe to hold the class IDs and the relative paths of the files
    df = pd.DataFrame.from_dict(
        filename_classid_dic, orient='index', columns=['class_id'])
    idx_col = np.arange(0, len(df), 1)
    df['idx_col'] = idx_col
    df['dir'] = df.index
    df.set_index('idx_col', inplace=True)
    print(df.head())

    dataset_name = os.path.basename(os.path.normpath(args.path))
    df_name = dataset_name + '.csv'
    df.to_csv(df_name, sep=',', header=True, index=False)


def make_data_dic(args):
    '''
    makes an imagefolder (imagenet style) with images of class in a certain
    folder into a txt dictionary with the first column being the
    file dir (relative) and the second into the class
    '''
    files_all = search_images(args)

    # filename and classid pairs
    filename_classid_dic = {}
    # id and class name/rel path as dict
    classid_classname_dic = {}

    idx = -1
    for fp in files_all:
        idx = add_image_to_dics(
            idx, fp, classid_classname_dic, filename_classid_dic)

    no_classes = idx + 1
    print('Total number of classes: ', no_classes)
    print('Total images files post-filtering (RGB only): ',
          len(filename_classid_dic))

    # save filename_classid df
    save_filename_classid(args, filename_classid_dic)

    # save classid_classname df
    save_classid_classname(classid_classname_dic)


def main():
    '''
    input is the path to the folder with imagenet-like structure
    imagenet/
    imagenet/class1/
    imagenet/class2/
    ...
    imagenet/classN/
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='path to folder like IN')
    args = parser.parse_args()

    make_data_dic(args)


main()
