import random
import numpy as np
import torch
from torchvision import transforms
from .utils_functions import set_seeds
from .kaggle_cifar_10_dataset import KaggleCIFAR10Dataset

set_seeds()


def load_dataset(cal_data_dir, cal_labels_dir, train_data_dir, train_labels_dir):
    """load dataset to datatloaders"""

    dataset = KaggleCIFAR10Dataset(
        cal_data_dir, 
        cal_labels_dir, 
         transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(),
                transforms.RandomRotation(10)
            ]))

    cal_dataloader, _ = dataset.get_train_val_dataloaders(
        0.999, 
        {
            'batch_size': 128,
            'shuffle': False,
        })
    
    dataset_train = KaggleCIFAR10Dataset(
        train_data_dir, 
        train_labels_dir, 
         transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(),
                transforms.RandomRotation(10)
            ]))

    train_dataloader, val_dataloader = dataset_train.get_train_val_dataloaders(
        0.9, 
        {
            'batch_size': 128,
            'shuffle': True,
        })
    
    return cal_dataloader, train_dataloader, val_dataloader