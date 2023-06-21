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

random.seed(123)
torch.manual_seed(123)

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
is_gpu = 1 if device == "cude:0" else 0


def read_images(path, n_images=2):
    images_paths=[]
    images=[]
    for root, _, files in os.walk(path):
        i=0
        for name in files:
            if i<n_images and name.endswith(".JPEG"):
                    image_path=os.path.join(root, name)
                    image=cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images_paths.append(image_path)
                    images.append(image)
                    i+=1        
    return images


def load_dataset(directory, batch_size, subset_size=100, if_subset=True):

    image_folder = datasets.ImageFolder(
        root=directory,
        transform=transforms.Compose([
            # transforms.ToTensor(),
            # transforms.Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(),
                transforms.RandomRotation(10),
                transforms.ToTensor()
        ])
    )
 
    subset_indices = []

    if if_subset:
        for class_index in range(len(image_folder.classes)):
            class_indices = np.where(np.array(image_folder.targets) == class_index)[0]
            
            if len(class_indices) >= subset_size:
                selected_indices = np.random.choice(class_indices, size=subset_size, replace=False)
                subset_indices.extend(selected_indices)

        image_folder = data.Subset(image_folder, subset_indices)
    dataloader = data.DataLoader(
        image_folder, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=2
    )

    return dataloader

def create_calibration_dataset(path, 
                               n_images=2, 
                               path_to_replace='./calibration_datasets_cifar10/calibration_datasets/cifar_10/train',
                               new_path='./cifar_10/train'):
   for root, dirs, files in os.walk(path):
        i=0
        for name in files:
            if i<n_images and name.endswith(".JPEG"):
                image_path=os.path.join(root, name)
                image=cv2.imread(image_path)
                image_name=image_path.split('\\')[-1]
                image_path=image_path.replace(path_to_replace, new_path)
                path_without_image_name=image_path.replace(image_name, '')
                Path(path_without_image_name).mkdir(parents=True, exist_ok=True)
                image_path=image_path.replace(image_name, 'calib_'+image_name)
                cv2.imwrite(image_path, image)
                i+=1
