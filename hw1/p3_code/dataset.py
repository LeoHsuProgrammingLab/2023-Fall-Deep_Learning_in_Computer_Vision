from torch.utils.data import Dataset
import os
from PIL import Image
from data_aug import *
import sys 
sys.path.append('..')
from mean_iou_evaluate import read_masks

class Semantic_Dataset(Dataset):
    def __init__(self, path, tfm = None, file_names = None):
        super(Semantic_Dataset).__init__()
        self.transform = tfm
        self.path = path
        self.file_names = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
        self.labels = read_masks(path)
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
        label = self.labels[idx]

        im = self.transform(im)
        label = self.transform(Image.fromarray(label))
        
        return im, label, fname.split('/')[-1]