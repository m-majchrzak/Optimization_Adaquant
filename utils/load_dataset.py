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

# data_directory='E:/DL/mini_imagenet_validation/val'
# path_to_replace='E:/DL/mini_imagenet_validation/val'
# new_path='E:/calibration_datasets/tiny_imagenet/train'

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


def load_dataset(directory, batch_size, subset_size=2, if_subset=True):

    image_folder = datasets.ImageFolder(
        root=directory,
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
                               path_to_replace='/kaggle_tiny_imagenet/tiny-imagenet-200/dataset_tiny_imagenet/train',
                               new_path='/calibration_datasets/tiny_imagenet/train'):
   for root, dirs, files in os.walk(path):
        i=0
        for name in files:
            if i<n_images and name.endswith(".JPEG"):
                image_path=os.path.join(root, name)
                image=cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_name=image_path.split('\\')[-1]
                image_path=image_path.replace(path_to_replace, new_path)
                path_without_image_name=image_path.replace(image_name, '')
                Path(path_without_image_name).mkdir(parents=True, exist_ok=True)
                image_path=image_path.replace(image_name, 'calib_'+image_name)
                cv2.imwrite(image_path, image)
                i+=1
