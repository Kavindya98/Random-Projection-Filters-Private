import torch
from torchvision import datasets, transforms
import numpy as np

def cal():

    dir_='./data/'
    train_dataset = datasets.SYN(
            dir_,transform=transforms.Compose([transforms.ToTensor()]), train=True, download=True)

    train = torch.utils.data.DataLoader(train_dataset, batch_size=10)
    
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for I, (batch_images,_) in enumerate(train):  # (B,C,H,W)
        print(batch_images.size())
        break
        channels_sum += torch.mean(batch_images, (0, 2, 3))
        channels_sqrd_sum += torch.mean(batch_images ** 2, (0, 2, 3))
        num_batches += 1

    # mean = channels_sum / num_batches
    # std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5
    # print('Mean ',mean)
    # print('Std ',std)

if __name__ == "__main__":
    cal()
