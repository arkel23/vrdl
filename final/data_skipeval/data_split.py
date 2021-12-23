import os
import argparse
import pandas as pd
import numpy as np


def data_split(args):
    # splits data into training and validation

    df = pd.read_csv(args.fn)
    print('Original df: ', len(df))

    n_per_class_df = df.groupby('class_id', as_index=True).count()

    df_list_train = []
    df_list_val = []
    for class_id, n_per_class in enumerate(n_per_class_df['dir']):
        train_samples_class = int(n_per_class*args.train)
        val_samples_class = n_per_class - train_samples_class
        assert(train_samples_class+val_samples_class == n_per_class)
        train_subset_class = df.loc[df['class_id'] == class_id].groupby(
            'class_id').head(train_samples_class)
        val_subset_class = df.loc[df['class_id'] == class_id].groupby(
            'class_id').tail(val_samples_class)
        df_list_train.append(train_subset_class)
        df_list_val.append(val_subset_class)

    df_train = pd.concat(df_list_train)
    df_val = pd.concat(df_list_val)

    print('Train df: ')
    print(df_train.head())
    print(df_train.shape)
    print('val df: ')
    print(df_val.head())
    print(df_val.shape)

    df_train_name = 'train.csv'
    df_train.to_csv(df_train_name, sep=',', header=True, index=False)

    df_val_name = 'val.csv'
    df_val.to_csv(df_val_name, sep=',', header=True, index=False)
    print('Finished saving train and val split dictionaries.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fn', type=str, help='path to data dic file')
    parser.add_argument('--train', type=float, default=0.9,
                        help='percent of data for training')
    parser.add_argument('--val', type=float, default=0.1,
                        help='percent of data for training')
    args = parser.parse_args()
    assert args.train + args.val == 1, 'Train + val ratios do not add to 1.'

    data_split(args)


main()
