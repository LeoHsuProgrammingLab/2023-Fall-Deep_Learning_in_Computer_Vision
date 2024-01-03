import json
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from torchvision.transforms import transforms
from tokenizer import BPETokenizer
from tqdm.auto import tqdm
from config import *

class ImageCaptionDataset(Dataset):
    def __init__(self, img_dir, json_file, tokenizer, tfm, max_tokens = config['max_tokens']):
        super().__init__()
        self.img_dir = img_dir
        print(img_dir)
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.tfm = tfm
        with open(json_file, 'r') as f:
            info = json.load(f)
        self.data = [
            {
                'caption': data['caption'],
                'img_id': data['image_id']
            } for data in info['annotations']
        ]
        # There are different captions with same img_id
        # temp = set()
        # for i in range(len(self.data)):
        #     temp.add(self.data[i]['img_id'])
        # print(len(temp))

        self.img_data = {
            img_data['id']: img_data['file_name'] for img_data in info['images']
        }

    def input_padding(self, cap_feats):
        input = cap_feats[:]
        input.insert(0, 50256)
        while len(input) < self.max_tokens:
            input.append(50256)
        return input

    def gt_padding(self, cap_feats):
        gt = cap_feats[:]
        gt.append(50256)
        while len(gt) < self.max_tokens:
            gt.append(-100)
        return gt

    # Because I return the unequal size of the data, I need to overwrite collate_fn
    # https://www.cnblogs.com/danieldaren/p/16594630.html
    def collate_fn(self, samples):
        img_list = []
        input_cap_id_list = []
        gt_cap_id_list = []
        cap_list = []
        fname_list = []
        for sample in samples:
            img_list.append(sample['img'])
            input_cap_id_list.append(sample['input_cap_ids'])
            gt_cap_id_list.append(sample['gt_cap_ids'])
            cap_list.append(sample['caption'])
            fname_list.append(sample['fname'])
        
        imgs = torch.stack(img_list, dim=0)
        # .t(): transpose
        input_cap_ids = torch.stack([torch.tensor(ids_list) for ids_list in input_cap_id_list], dim=0)
        gt_cap_ids = torch.stack([torch.tensor(ids_list) for ids_list in gt_cap_id_list], dim=0)
        return {
            'img': imgs, 
            'input_cap_ids': input_cap_ids,
            'gt_cap_ids': gt_cap_ids,
            'caption': cap_list,
            'fname': fname_list # for output json file
        }

    def __getitem__(self, index):
        cap_imgid_pair = self.data[index]
        cap = cap_imgid_pair['caption']
        cap_feats = self.tokenizer.encode(cap)
        input_cap_feats = self.input_padding(cap_feats)
        gt_cap_feats = self.gt_padding(cap_feats)
        img_name = self.img_data[cap_imgid_pair['img_id']]
        img = Image.open(self.img_dir + '/' + img_name).convert('RGB')
        img = self.tfm(img)

        return {
            'img': img, 
            'input_cap_ids': input_cap_feats,
            'gt_cap_ids': gt_cap_feats,
            'caption': cap,
            'fname': os.path.splitext(img_name)[0] # for output json file
        }

    def __len__(self):
        return len(self.data)
    
class ImageDataset(Dataset): # for inference
    def __init__(self, img_dir, tfm):
        super().__init__()
        self.imgs = [os.path.join(img_dir, x) for x in os.listdir(img_dir) if x.endswith('jpg')]
        self.fnames = [x for x in os.listdir(img_dir) if x.endswith('jpg')]
        self.tfm = tfm
    
    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert('RGB')
        print(img)
        img = self.tfm(img)
        fname = self.fnames[idx].replace('.jpg', '')
        return img, fname
    
    def __len__(self):
        return len(self.imgs)


    
if __name__ == "__main__":
    tfm = transforms.Compose([
        transforms.ToTensor()
    ])

    train_set = ImageCaptionDataset(
        img_dir='/home/leohsu-cs/DLCV2023/dlcv-fall-2023-hw3-LeoHsuProgrammingLab/hw3_data/p2_data/images/train',
        json_file='/home/leohsu-cs/DLCV2023/dlcv-fall-2023-hw3-LeoHsuProgrammingLab/hw3_data/p2_data/train.json',
        tokenizer=BPETokenizer(
            encoder_file = '/home/leohsu-cs/DLCV2023/dlcv-fall-2023-hw3-LeoHsuProgrammingLab/encoder.json',
            vocab_file = '/home/leohsu-cs/DLCV2023/dlcv-fall-2023-hw3-LeoHsuProgrammingLab/vocab.bpe'
        ), 
        tfm=tfm
    )

    val_set_ = ImageCaptionDataset(
        img_dir='/home/leohsu-cs/DLCV2023/dlcv-fall-2023-hw3-LeoHsuProgrammingLab/hw3_data/p2_data/images/val',
        json_file='/home/leohsu-cs/DLCV2023/dlcv-fall-2023-hw3-LeoHsuProgrammingLab/hw3_data/p2_data/val.json',
        tokenizer=BPETokenizer(
            encoder_file = '/home/leohsu-cs/DLCV2023/dlcv-fall-2023-hw3-LeoHsuProgrammingLab/encoder.json',
            vocab_file = '/home/leohsu-cs/DLCV2023/dlcv-fall-2023-hw3-LeoHsuProgrammingLab/vocab.bpe'
        ), 
        tfm=tfm
    )

    val_set = ImageDataset(
        img_dir='/home/leohsu-cs/DLCV2023/dlcv-fall-2023-hw3-LeoHsuProgrammingLab/hw3_data/p2_data/images/val', 
        tfm=tfm
    )
    print('val_set:', len(val_set))
    print(train_set[0])
    num = 0
    for i in tqdm(range(len(train_set))):
        length = len(train_set[i]['caption'].split())
        # assert(len(train_set[i]['input_cap_ids'])==len(train_set[i]['gt_cap_ids']))
        if num < length:
            num = length

    for i in tqdm(range(len(val_set_))):
        length = len(val_set_[i]['caption'].split())
        # assert(len(val_set_[i]['input_cap_ids'])==len(val_set_[i]['gt_cap_ids']))
        if num < length:
            num = length
    print('the most tokens: ', num) # 47 tokens
    
