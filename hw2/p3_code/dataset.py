import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset
from data_aug import train_tfm, test_tfm


class DigitsDataset(Dataset):
    def __init__(self, path, transform, csv):
        super().__init__()
        self.tfm = transform
        self.img_names = []
        self.labels = []

        df = pd.read_csv(csv)
        for row in df.iterrows():
            self.img_names.append(path + row[1][0]) # name
            self.labels.append(row[1][1]) # label
        
        print(f"load_set with {len(self.img_names)} images")
    
    def __getitem__(self, idx):
        img = Image.open(self.img_names[idx]).convert('RGB')
        img = self.tfm(img)
        print(img.shape)
        label = self.labels[idx]

        return img, label
    
    def __len__(self):
        return len(self.img_names)

if __name__=="__main__":

    domains = ['mnistm', 'svhn', 'usps']
    domain = domains[2]
    img_path = f'/home/leohsu-cs/DLCV2023/hw2-LeoHsuProgrammingLab/hw2_data/digits/{domain}/data/'
    train_csv = f'/home/leohsu-cs/DLCV2023/hw2-LeoHsuProgrammingLab/hw2_data/digits/{domain}/train.csv'
    test_csv = f'/home/leohsu-cs/DLCV2023/hw2-LeoHsuProgrammingLab/hw2_data/digits/{domain}/test.csv'
    val_csv = f'/home/leohsu-cs/DLCV2023/hw2-LeoHsuProgrammingLab/hw2_data/digits/{domain}/val.csv'

    train_set = DigitsDataset(
        path = img_path, 
        transform = train_tfm,
        csv = train_csv
    )

    test_set = DigitsDataset(
        path = img_path,
        transform = test_tfm, 
        csv = test_csv
    )

    val_set = DigitsDataset(
        path = img_path,
        transform = test_tfm,
        csv = val_csv
    )

    print(len(train_set), len(test_set), len(val_set))
    print(train_set[0], test_set[0], val_set[0])