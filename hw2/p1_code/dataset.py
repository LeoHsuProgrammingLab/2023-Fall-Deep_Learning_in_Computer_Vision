from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

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
    
    def __getitem__(self, idx):
        img = Image.open(self.img_names[idx]).convert('RGB')
        img = self.tfm(img)
        label = self.labels[idx]

        return img, label
    
    def __len__(self):
        return len(self.img_names)

if __name__=="__main__":
    dataset = DigitsDataset(
        path = '/home/leohsu-cs/DLCV2023/hw2-LeoHsuProgrammingLab/hw2_data/digits/mnistm/data/', 
        transform = None,
        csv = '/home/leohsu-cs/DLCV2023/hw2-LeoHsuProgrammingLab/hw2_data/digits/mnistm/train.csv'
    )

    print(len(dataset))