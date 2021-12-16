import os
import argparse
import pandas as pd
import numpy as np


def data_split(args):
    # splits data into training and testing

    df = pd.read_csv(args.fn)
    print('Original df: ', len(df))

    n_per_class_df = df.groupby('class_id', as_index=True).count()

    df_list_train = []
    df_list_test = []
    for class_id, n_per_class in enumerate(n_per_class_df['dir']):
        train_samples_class = int(n_per_class*args.train)
        test_samples_class = n_per_class - train_samples_class
        assert(train_samples_class+test_samples_class == n_per_class)
        train_subset_class = df.loc[df['class_id'] == class_id].groupby(
            'class_id').head(train_samples_class)
        test_subset_class = df.loc[df['class_id'] == class_id].groupby(
            'class_id').tail(test_samples_class)
        df_list_train.append(train_subset_class)
        df_list_test.append(test_subset_class)

    df_train = pd.concat(df_list_train)
    df_test = pd.concat(df_list_test)

    print('Train df: ')
    print(df_train.head())
    print(df_train.shape)
    print('Test df: ')
    print(df_test.head())
    print(df_test.shape)

    df_train_name = 'train.csv'
    df_train.to_csv(df_train_name, sep=',', header=True, index=False)

    df_test_name = 'test.csv'
    df_test.to_csv(df_test_name, sep=',', header=True, index=False)
    print('Finished saving train and test split dictionaries.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fn', type=str, help='path to data dic file')
    parser.add_argument('--train', type=float, default=0.9,
                        help='percent of data for training')
    parser.add_argument('--test', type=float, default=0.1,
                        help='percent of data for training')
    args = parser.parse_args()
    assert args.train + args.test == 1, 'Train + test ratios do not add to 1.'

    data_split(args)


main()
