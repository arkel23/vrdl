import os

import pandas as pd
from PIL import Image
import torch
import torch.utils.data as data


class Fish(data.Dataset):
    '''
    8 classes:
    ALB,0
    BET,1
    DOL,2
    LAG,3
    NoF,4
    OTHER,5
    SHARK,6
    YFT,7
    '''
    def __init__(self, root, train=True, transform=None):
        super().__init__()
        self.root = os.path.abspath(os.path.join(root))
        self.train = train
        self.transform = transform

        if self.train:
            self.set_dir = os.path.join(self.root, 'train.csv')
        else:
            self.set_dir = os.path.join(self.root, 'val.csv')
        self.df = pd.read_csv(self.set_dir)

        self.targets = self.df['class_id'].to_numpy()
        self.data = self.df['dir'].to_numpy()

        # classid_classname_df for mapping between ids and names and viceversa
        class_mapper_path = os.path.join(self.root, 'classid_classname.csv')
        self.classes = pd.read_csv(class_mapper_path)
        self.num_classes = len(self.classes)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_dir, target = self.data[idx], self.targets[idx]
        img_dir = os.path.join(self.root, 'fish', img_dir)
        img = Image.open(img_dir)

        if self.transform:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.targets)


if __name__ == '__main__':
    dataset = Fish('./data/', train=True)
    print(len(dataset), dataset[0])
    dataset = Fish('./data/', train=False)
    print(len(dataset), dataset[0])
