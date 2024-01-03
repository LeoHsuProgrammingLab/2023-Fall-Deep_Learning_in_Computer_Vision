from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch
import os
import argparse
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import pandas as pd


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
            
        return im,label, fname.split("/")[-1]

parser = argparse.ArgumentParser()
parser.add_argument("-t", type=str)
parser.add_argument("-o", type=str)
args = parser.parse_args()
test_path = args.t
output_path = args.o
print('testing_images_dir_path', test_path)
print('output_csv_path', output_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
])

test_set = P1_Dataset(test_path, transform)
test_loader = DataLoader(test_set, batch_size = 32, shuffle = False)

resnext101 = models.resnext101_64x4d(weights = None)
resnext101.fc = nn.Sequential(
    nn.Linear(in_features= resnext101.fc.in_features, out_features = 50),
)

model = resnext101.to(device)
model_path = "./P1_model_B.ckpt"
model.load_state_dict(torch.load(model_path))
model.eval()

output = []
for batch in tqdm(test_loader):
    imgs = batch[0]
    imgs = imgs.to(device)
    file_name = batch[2]
    logits = model(imgs)
    pred = logits.argmax(dim=-1).cpu().numpy()
    for i in range(len(pred)):
        output.append([file_name[i], str(pred[i])])

df = pd.DataFrame(output, columns = ('filename', 'label'))
df.to_csv(output_path, index = True)  