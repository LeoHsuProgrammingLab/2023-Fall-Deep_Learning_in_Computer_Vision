import torch
import argparse
import os
import pandas as pd
import torch.nn as nn
from tqdm.auto import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

class P2_office(Dataset):
    def __init__(self, path, tfm = None, file_names = None):
        super(P2_office).__init__()
        self.transform = tfm
        self.path = path
        self.file_names = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
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
            
        return im,label, fname.split("/")[-1]
    
class MyResnet50(nn.Module):
    def __init__(self, backbone_ckpt = None, setting = "C"):
        super().__init__()
        self.resnet50 = models.resnet50(weights = None)
        if backbone_ckpt != None:
            self.resnet50.load_state_dict(torch.load(backbone_ckpt))
        if setting == "D" or setting == "E":
            for param in self.resnet50.parameters():
                param.requires_grad = False
        
        self.fc = nn.Sequential(
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 65)
        )

    def forward(self, x):
        x = self.resnet50(x)
        x = self.fc(x)
        return x

parser = argparse.ArgumentParser()
parser.add_argument("-c", type=str)
parser.add_argument("-t", type=str)
parser.add_argument("-o", type=str)
args = parser.parse_args()
input_csv_path = args.c
test_path = args.t
output_path = args.o
print('test_images_dir_path', test_path)
print('output_csv_path', output_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std  = [0.229, 0.224, 0.225]),
])

test_set = P2_office(path = test_path, tfm=tfm)
test_loader = DataLoader(test_set, batch_size = 32, shuffle = False)

model = MyResnet50(backbone_ckpt = "./P2_resnet50.ckpt", setting = "C")
model.load_state_dict(torch.load("./P2_classifier_C.ckpt"))
model = model.to(device)

model.eval()
output = []
for batch in tqdm(test_loader):
    imgs = batch[0].to(device)
    file_name = batch[2]
    logits = model(imgs)
    pred = logits.argmax(dim=-1).cpu().numpy()
    for i in range(len(pred)):
        output.append([file_name[i], str(pred[i])])

df = pd.DataFrame(output, columns = ('filename', 'label'))
df.to_csv(output_path, index=True)  