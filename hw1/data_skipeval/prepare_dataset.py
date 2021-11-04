import argparse
import pandas as pd


def create_categorical():
    df = pd.read_csv('training_labels.txt', sep=' ', index_col=False,
                     header=None, names=['file_name', 'class_name'],
                     dtype={'file_name': 'object', 'class_name': 'category'})

    df['class_id'] = df['class_name'].cat.codes

    df_cat = df.copy()
    df_cat = df_cat[['file_name', 'class_id']]
    df_cat.to_csv('train_val.csv', header=True, index=False)

    df_class_dic = df.copy()
    df_class_dic = df_class_dic[['class_id', 'class_name']]
    df_class_dic = df_class_dic.drop_duplicates()
    df_class_dic.to_csv('id2name_dic.csv', header=True, index=False)


def split_train_val(split_percent):
    df = pd.read_csv('train_val.csv')
    print('Original train-val split\n', df.head(), '\n', df.shape)

    samples_per_class_df = df.groupby('class_id', as_index=True).count()

    df_list_train = []
    df_list_val = []
    for class_id, total_samples_class in enumerate(
            samples_per_class_df['file_name']):
        train_samples_class = int(total_samples_class*split_percent[0])
        val_samples_class = total_samples_class - train_samples_class
        assert(train_samples_class+val_samples_class == total_samples_class)
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
    print('Val df: ')
    print(df_val.head())
    print(df_val.shape)

    df_train_name = 'train.csv'
    df_train.to_csv(df_train_name, sep=',', header=True, index=False)

    df_val_name = 'val.csv'
    df_val.to_csv(df_val_name, sep=',', header=True, index=False)
    print('Finished saving train and val split dictionaries.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_percent', type=float, default=0.7)
    parser.add_argument('--val_percent', type=float, default=0.3)
    args = parser.parse_args()

    split_percent = [args.train_percent, args.val_percent]

    create_categorical()
    split_train_val(split_percent)


if __name__ == '__main__':
    main()
