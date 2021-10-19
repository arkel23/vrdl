import os

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.utils.data as data


class Birds(data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
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
    

class BirdsInstance(Birds):
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
            
        return img, target, idx


class BirdsInstanceSample(Birds):
    def __init__(self, root,
                 transform=None, target_transform=None,
                 k=4096, mode='exact', is_sample=True, percent=1.0):
        super().__init__(root=root, transform=transform, 
                         target_transform=target_transform)
        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        num_classes = self.num_classes
        num_samples = len(self.targets)
        label = self.targets
        
        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

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

        if not self.is_sample:
            # directly return
            return img, target, idx
        else:
            # sample contrastive examples
            if self.mode == 'exact':
                pos_idx = idx
            elif self.mode == 'relax':
                pos_idx = np.random.choice(self.cls_positive[target], 1)
                pos_idx = pos_idx[0]
            else:
                raise NotImplementedError(self.mode)
            replace = True if self.k > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, idx, sample_idx
        

if __name__ == '__main__':
    dataset = Birds(root='../../data/')
    print(dataset, len(dataset), dir(dataset), dataset.__getitem__(0))
    
    dataset = BirdsInstance(root='../../data/')
    print(dataset, len(dataset), dir(dataset), dataset.__getitem__(0))

    dataset = BirdsInstanceSample(root='../../data/')
    print(dataset, len(dataset), dir(dataset), dataset.__getitem__(0))