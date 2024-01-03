# Output: JSON file (key: filename without .jpg)
# CIDEr: Similarity from sentences
# CLIPScore: Correspondence of images and texts
# Hand in only one (the best) setting from 3 implementation
# model parameters need to be < 35M
# Load checkpoint with "strict=False"
# print("Total params:", sum(p,numel() for p in model.parameters() if p.requires_grad))
# Visualize the cross-attention block feature map  

import torch
import json
import timm
import argparse
from tqdm.auto import tqdm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from utils import fixed_init
from config import config
from model import *
from dataset import *
from torch.utils.data import DataLoader

def inference(model, model_path, test_loader, tokenizer, json_output_path, device):
    model = model.to(device)
    state_dict = torch.load(model_path)
    print(sum([p.numel() for n, p in state_dict.items()]))
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    preds_img_sentence = {}
    with torch.no_grad():
        for batch in tqdm(test_loader): # set batch size = 1
            # print(batch[0])
            output_ids = model.greedy(batch[0].to(device))
            output_sentence = tokenizer.decode(output_ids)
            # print(output_sentence)
            preds_img_sentence[str(batch[1][0])] = output_sentence
    
    with open(json_output_path, 'w') as json_f:
        json.dump(preds_img_sentence, json_f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", type=str) #test images
    parser.add_argument("-o", type=str) #json file
    parser.add_argument("-d", type=str) #decoder weights
    args = parser.parse_args()

    test_imgs_path = args.t
    output_path = args.o
    decoder_weights = args.d

    fixed_init(666)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vits = [
        'vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k', 
        'vit_large_patch14_clip_336.openai_ft_in12k_in1k'
    ]
    encoder = timm.create_model(vits[1], pretrained=True)
    encode_transform = create_transform(**resolve_data_config(encoder.pretrained_cfg, model=encoder))
    # print(encoder.feature_info)
    # print(encoder)

    cfg = Config(checkpoint = decoder_weights)
    decoder = Decoder(cfg)

    tokenizer = BPETokenizer(
        encoder_file = '/home/leohsu-cs/DLCV2023/dlcv-fall-2023-hw3-LeoHsuProgrammingLab/encoder.json',
        vocab_file = '/home/leohsu-cs/DLCV2023/dlcv-fall-2023-hw3-LeoHsuProgrammingLab/vocab.bpe'
    )

    model = ImageCaptionTransformer(
        decoder = decoder, 
        encoder = encoder
    )

    val_set = ImageDataset(
        img_dir = test_imgs_path, 
        tfm = encode_transform
    )

    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=True, num_workers=3, worker_init_fn=fixed_init(666)
    )

    inference(
        model, 
        '/home/leohsu-cs/DLCV2023/dlcv-fall-2023-hw3-LeoHsuProgrammingLab/p2_code/ckpt/1best_PT_e5.ckpt', 
        val_loader, 
        tokenizer, 
        output_path, 
        device
    )