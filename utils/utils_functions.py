import random
import numpy as np
import torch

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