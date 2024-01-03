import os
import imageio
import argparse
import torch
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights,
)
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, FCNHead


def read_masks(filepath):
    '''
    Read masks from directory and transform to categorical
    '''
    file_list = [file for file in os.listdir(filepath) if file.endswith('.png')]
    file_list.sort()
    n_masks = len(file_list)
    masks = np.empty((n_masks, 512, 512))

    for i, file in enumerate(file_list):
        mask = imageio.imread(os.path.join(filepath, file))
        mask = (mask >= 128).astype(int)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2] # Combine R, G, B with different weights
        masks[i, mask == 3] = 0  # (Cyan: 011) Urban land 
        masks[i, mask == 6] = 1  # (Yellow: 110) Agriculture land 
        masks[i, mask == 5] = 2  # (Purple: 101) Rangeland 
        masks[i, mask == 2] = 3  # (Green: 010) Forest land 
        masks[i, mask == 1] = 4  # (Blue: 001) Water 
        masks[i, mask == 7] = 5  # (White: 111) Barren land 
        masks[i, mask == 0] = 6  # (Black: 000) Unknown 

    return masks

def pred2image(pred, name, out_path):
    pred = pred.numpy()
    pred_img = np.zeros((512, 512, 3), dtype=np.uint8)
    pred_img[pred == 0] = [0, 255, 255]
    pred_img[pred == 1] = [255, 255, 0]
    pred_img[pred == 2] = [255, 0, 255]
    pred_img[pred == 3] = [0, 255, 0]
    pred_img[pred == 4] = [0, 0, 255]
    pred_img[pred == 5] = [255, 255, 255]
    pred_img[pred == 6] = [0, 0, 0]
    imageio.imwrite(os.path.join(out_path, name.replace('sat.jpg', f'mask.png')), pred_img)

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
    
parser = argparse.ArgumentParser() 
parser.add_argument("-t", type=str)
parser.add_argument("-o", type=str)
args = parser.parse_args()
test_path = args.t
output_dir = args.o

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tfm = transforms.Compose([
    transforms.ToTensor(),
])

test_set = Semantic_Dataset(path = test_path, tfm = tfm)
test_loader = DataLoader(test_set, batch_size = 5, shuffle = False)

model = deeplabv3_resnet50(weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1, weights_backbone = ResNet50_Weights.DEFAULT)
model.aux_classifier[4] = nn.Conv2d(256, 7, kernel_size = 1, stride = 1)
model.classifier[4] = nn.Conv2d(256, 7, kernel_size = 1, stride = 1)
model = model.to(device)
model.load_state_dict(torch.load('./P3_model_B.ckpt'))
model.eval()

for batch in tqdm(test_loader):
    imgs = batch[0].to(device)
    file_name = batch[2]
    logits = model(imgs)['out']
    pred_masks = logits.argmax(dim = 1).cpu()
    for i in range(len(file_name)):
        pred2image(pred_masks[i], file_name[i], output_dir)
