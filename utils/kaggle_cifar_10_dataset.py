import os
import torch
import numpy as np
import pandas as pd
from torchvision import datasets
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import convert_image_dtype, rgb_to_grayscale
import matplotlib.pyplot as plt

class KaggleCIFAR10Dataset(Dataset):

    def __init__(self, img_dir, labels_file, transform = None, target_transform = None, convert_to_float = True) -> None:
        super().__init__()
        self.img_dir = img_dir
        self.img_names = [e for e in os.listdir(img_dir) if e.endswith('.png')]
        self.img_labels = pd.read_csv(labels_file, index_col = 0)
        self.transform = transform
        self.labels_mapping = self.get_labels_mapping()
        self.target_transform = self.labels_mapping if target_transform is None else target_transform
        self.convert_to_float = convert_to_float

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, index):
        img_name = self.img_names[index]
        pandas_index = int(img_name.split('.')[0])
        img_path = os.path.join(self.img_dir, img_name)
        image = read_image(img_path)
        if self.convert_to_float: image = convert_image_dtype(image)
        label = self.img_labels.loc[pandas_index].values[0]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform[label]
        return image, label

    def get_labels_mapping(self):
        uniq_labels = self.img_labels['label'].unique()
        return {label: idx for idx, label in enumerate(uniq_labels)}

    def get_train_val_dataloaders(self, train_fraction, dataloader_kwargs):
        train_val_datasets = random_split(self, (train_fraction, 1 - train_fraction))
        dataloaders = []
        for dataset in train_val_datasets:
            dataloaders.append(DataLoader(dataset, **dataloader_kwargs))
        return dataloaders

class KaggleCIFAR10TestDataset(Dataset):

    def __init__(self, img_dir, transform = None, target_transform = None, convert_to_float = True):
        super().__init__()
        self.img_dir = img_dir
        self.img_names_sorted = self.get_sorted_img_ids()
        self.transform = transform
        self.target_transform = target_transform
        self.convert_to_float = convert_to_float

    def __len__(self):
        return len(self.img_names_sorted)
    
    def __getitem__(self, index):
        img_name = self.img_names_sorted[index]
        img_path = os.path.join(self.img_dir, img_name)
        image = read_image(img_path)
        if self.convert_to_float: image = convert_image_dtype(image)
        if self.transform:
            image = self.transform(image)
        return image, index, img_name

    def get_sorted_img_ids(self):
        img_names = np.array([name for name in os.listdir(self.img_dir) if name.endswith('png')])
        img_names_numeric = [int(e.split('.')[0]) for e in img_names]
        sorting_perm = np.argsort(img_names_numeric)
        return img_names[sorting_perm]
    
    def get_dataloader(self, **kwargs):
        return DataLoader(self, **kwargs)

if __name__ == '__main__':
    img_dir = 'cifar-10/test'
    dataset = KaggleCIFAR10TestDataset(img_dir)
    dataloader_kwargs = {
        'shuffle': False,
        'batch_size': 2
    }
    dataloader = dataset.get_dataloader(**dataloader_kwargs)
    for batch in dataloader:
        imgs, idxs, names = batch
        print(idxs, names)
        break

