# Import necessary packages.
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, random_split
from data_aug import train_tfm_A

def train_valid_split(data_set, valid_ratio, seed):
    '''Split provided training data into training set and validation set'''
    valid_set_size = int(valid_ratio * len(data_set)) 
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    return train_set, valid_set

class P1_Dataset(Dataset):
    def __init__(self, path, tfm = None, file_names = None):
        super(P1_Dataset).__init__()
        self.transform = tfm
        self.path = path
        self.file_names = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".png")])
        if file_names != None:
            self.file_names = file_names
            print(f"One {path} sample",self.file_names)
        else:
            print(f"One {path} sample",self.file_names[0])
  
    def __len__(self):
        return len(self.file_names)
  
    def __getitem__(self,idx):
        fname = self.file_names[idx]
        im = Image.open(fname)
        im = self.transform(im)
        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -100
            
        return im,label
    
def cal_dataset_mean_std(dataset):
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for img, label in dataset:
        mean += img.mean([1,2]) # along H, W means we sum over C (C, H, W)
        std += img.std([1,2])
    mean /= len(dataset)
    std /= len(dataset)
    return mean, std

if __name__ == "__main__":
    dataset = P1_Dataset("../hw1_data/p1_data/train_50", train_tfm_A)
    mean, std = cal_dataset_mean_std(dataset)
    print(f"mean = {mean}", f"std = {std}")
