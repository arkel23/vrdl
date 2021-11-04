import os

import pandas as pd
from PIL import Image
import torch
import torch.utils.data as data


class Birds(data.Dataset):
    def __init__(self, root, train=True,
                 transform=None, target_transform=None):
        self.root = os.path.abspath(root)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if self.train:
            labels_path = os.path.join(root, 'train.csv')
        else:
            labels_path = os.path.join(root, 'val.csv')
        id2name_path = os.path.join(root, 'id2name_dic.csv')

        df = pd.read_csv(labels_path)
        id2name_dic = pd.read_csv(id2name_path)

        self.data = df['file_name'].to_numpy()
        self.targets = df['class_id'].to_numpy()

        self.classes = id2name_dic
        self.num_classes = len(self.classes)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root, 'train', self.data[idx])
        img = Image.open(img_path).convert('RGB')
        target = self.targets[idx]

        if self.transform:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.targets)


if __name__ == '__main__':
    dataset = Birds(root='../../data/')
    print(dataset, len(dataset), dir(dataset), dataset.__getitem__(0))
