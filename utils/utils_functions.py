import os
import cv2
import random
import numpy as np
np.random.seed(123)
import torch
from torch.utils import data
from pathlib import Path

from torchvision import datasets
from torchvision import transforms

from .kaggle_cifar_10_dataset import KaggleCIFAR10Dataset

random.seed(123)
torch.manual_seed(123)

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
is_gpu = 1 if device == "cude:0" else 0


def load_dataset(cal_data_dir, cal_labels_dir, train_data_dir, train_labels_dir):
    dataset = KaggleCIFAR10Dataset(
        cal_data_dir, 
        cal_labels_dir, 
         transforms.Compose([ # basic augmentation
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
         transforms.Compose([ # basic augmentation
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

def val_loop(dataloader, model, device):
    size = len(dataloader.dataset)
    score=0
    with torch.no_grad():
        for batch_imgs, batch_labels in dataloader:
            batch_imgs, batch_labels = batch_imgs.to(device), batch_labels.to(device)
            logits = model(batch_imgs)
            score += (logits.argmax(1) == batch_labels).type(torch.float).sum().item()
    score /= size
    accuracy = 100 * score
    return accuracy

def set_seeds():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)