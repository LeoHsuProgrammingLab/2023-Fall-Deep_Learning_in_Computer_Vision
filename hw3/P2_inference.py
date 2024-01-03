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
import numpy as np
import random
import math
import torch.nn.functional as F
import collections
import loralib as lora
import re
import os
from torch.utils.data import Dataset
from PIL import Image
from tqdm.auto import tqdm
from torch import nn, Tensor
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader

config = {
    'n_epochs': 10, 
    'batch_size': 32, 
    'lr': 5e-4, 
    'patience': 10, 
    'weight_decay': 1e-2, 
    'max_tokens': 60, 
    'setting': 'adapter'
}

class BPETokenizer:
    
    def __init__(self, encoder_file, vocab_file):
        with open(encoder_file, 'r', encoding='utf-8') as f:
            self.encoder = json.load(f)
        self.decoder = {v:k for k,v in self.encoder.items()}
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab = f.read().split('\n')[1:-1]
        self.bpe_ranks = {tuple(line.split()): i for i, line in enumerate(vocab)}
        assert len(self.encoder) == 50257 and len(self.bpe_ranks) == 50000
        bs = list(range(33, 127)) + list(range(161, 256))
        xs = list(range(0, 33)) + list(range(127, 161))
        cs = bs[:] + [2**8 + i for i in range(len(xs))]
        self.byte_encoder = dict(zip(bs + xs, [chr(n) for n in cs]))
        self.byte_decoder = {v:k for k, v in self.byte_encoder.items()}

    def encode(self, text, allowed_special=None):
        tokens = re.findall(r"""<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d| ?""" +
                            r"""\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+""", text, re.UNICODE)
        def translate(token):
            if token == '<|endoftext|>':
                assert allowed_special and token in allowed_special
                return [token]
            word = tuple(''.join(self.byte_encoder[byte] for byte in token.encode('utf-8')))
            while len(word) != 1:
                pairs = set((word[i], word[i+1]) for i in range(len(word)-1))
                bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
                if bigram not in self.bpe_ranks:
                    break
                a, b = bigram
                new_word = []
                i = 0
                while i < len(word):
                    j = word.index(a, i) if a in word[i:] else len(word)
                    new_word.extend(word[i:j])
                    i = j
                    if i < len(word):
                        j = 2 if i < len(word)-1 and word[i] == a and word[i+1] == b else 1
                        new_word.append(a+b if j == 2 else word[i])
                        i += j
                word = tuple(new_word)
            return word
        return [self.encoder[_] for token in tokens for _ in translate(token)]

    def decode(self, tokens):
        tokens = [self.decoder[token] for token in tokens]
        buffer = bytearray([self.byte_decoder[c] for c in ''.join(tokens)])
        return buffer.decode('utf-8', errors='replace')


class Config:

    def __init__(self, checkpoint=None):
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768
        self.vocab_size = 50257
        self.block_size = 1024
        self.checkpoint = checkpoint

class Attention(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        size = cfg.block_size
        self.register_buffer('bias', torch.tril(torch.ones(size, size)).view(1, 1, size, size))

    def forward(self, x):
        B, T, C = x.size() # batch, context, embedding
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))

class CrossAttention(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.c_q = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.c_k = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.c_v = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        # size = cfg.block_size
        # self.register_buffer('bias', torch.tril(torch.ones(size, size)).view(1, 1, size, size))

    def forward(self, x_text_feats, x_img_feats):
        # I want q from texts, k, v from images, 
        B, T_t, C_t = x_text_feats.size() # batch, context, n_embd
        _, T_i, C_i = x_img_feats.size() # batch, context, n_embd
        q = self.c_q(x_text_feats).view(B, T_t, self.n_head, C_t // self.n_head).transpose(1, 2)
        k = self.c_k(x_img_feats).view(B, T_i, self.n_head, C_i // self.n_head).transpose(1, 2)
        v = self.c_v(x_img_feats).view(B, T_i, self.n_head, C_i // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # att = att.masked_fill(self.bias[:,:,:T_t,:T_i] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T_t, C_t)), att

class Adapter(nn.Module):

    def __init__(self, n_embd, n_latent = 256):
        super().__init__()
        self.down = nn.Linear(n_embd, n_latent)
        self.non_linear = nn.GELU(approximate='tanh')
        self.up = nn.Linear(n_latent, n_embd)

    def forward(self, x):
        x = self.down(x)
        x = self.non_linear(x)
        x = self.up(x)
        return x
    
class PrefixTuner(nn.Module):

    def __init__(self, n_embd, prefix_len = 10):
        super().__init__()
        self.len = prefix_len
        self.prefix = nn.Parameter(torch.randn(config['batch_size'], prefix_len, n_embd))

    def forward(self, x):
        prefix = self.prefix[:x.shape[0], :, :]
        extended_x = torch.cat([prefix, x], dim=1)

        return extended_x

class Block(nn.Module):

    def __init__(self, cfg, with_adapter = False, with_lora = False, with_prefix_tuning = False):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.cross_attn = CrossAttention(cfg)
        self.mlp = nn.Sequential(collections.OrderedDict([
            ('c_fc', 
                nn.Linear(cfg.n_embd, 4 * cfg.n_embd) if not with_lora else 
                lora.Linear(cfg.n_embd, 4 * cfg.n_embd, r = 8)
            ),
            ('act', nn.GELU(approximate='tanh')),
            ('c_proj', 
                nn.Linear(4 * cfg.n_embd, cfg.n_embd) if not with_lora else
                lora.Linear(4 * cfg.n_embd, cfg.n_embd, r = 8)
            )
        ]))
        self.with_adapter = with_adapter
        self.with_PT = with_prefix_tuning
        self.adapter = Adapter(cfg.n_embd)
        self.prefix_len = 15
        self.PT = PrefixTuner(cfg.n_embd, self.prefix_len)

    def forward(self, x, x_img_feats):
        # For prefix tuning only
        if self.with_PT:
            x = self.PT(x)

        x = x + self.attn(self.ln_1(x))
        # print('s attn', x.shape)
        cross_attn_feats, cross_attn_heats = self.cross_attn(self.ln_1(x), x_img_feats)
        x = x + cross_attn_feats

        # For adapter only
        if self.with_adapter:
            x = x + self.adapter(self.mlp(self.ln_2(x))) # add adapter and lora here
        else:
            x = x + self.mlp(self.ln_2(x))
            
        # For prefix tuning only
        if self.with_PT:
            x = x[:, self.prefix_len:, :]

        return x

# Add cross-attention & PEFT to Decoder
# 1. Adapter: Design dimensions & the position of adapter
# 2. Prefix tuning, Prompt tuning, P-tuning
# 3. Lora: loralib, github for LoRA
class Decoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.PEFT_layer_num = 2
        self.h_dict = {
            "h": nn.Sequential(
                *[Block(cfg) for _ in range(cfg.n_layer)]
            ),
            "h_PT": nn.Sequential(
                *[Block(cfg, with_prefix_tuning=True) for _ in range(cfg.n_layer)]
            ),
            "h_adapter": nn.Sequential(
                *[Block(cfg) for _ in range(cfg.n_layer - self.PEFT_layer_num)],
                *[Block(cfg, with_adapter=True) for _ in range(self.PEFT_layer_num)]
            ),
            "h_lora" : nn.Sequential(
                *[Block(cfg) for _ in range(cfg.n_layer - self.PEFT_layer_num)],
                *[Block(cfg, with_lora=True) for _ in range(self.PEFT_layer_num)]
            ),
        }
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe = nn.Embedding(cfg.block_size, cfg.n_embd),
            # Damn nn.Sequential can only have one input
            h = self.h_dict['h_PT'],
            ln_f = nn.LayerNorm(cfg.n_embd)
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        # load checkpoint
        if self.cfg.checkpoint is not None:
            state_dict = torch.load(self.cfg.checkpoint)
            transposed = [ '.c_attn.weight', '.c_fc.weight', '.c_proj.weight' ]
            for key, value in state_dict.items():
                if any(key.endswith(w) for w in transposed):
                    state_dict[key] = value.t()
            self.transformer.load_state_dict(state_dict, strict=False)

    def forward(self, x_text_feats: Tensor, x_img_feats: Tensor):
        x_text_feats = torch.narrow(x_text_feats, 1, 0, min(x_text_feats.size(1), self.block_size))
        pos = torch.arange(x_text_feats.size()[1], dtype=torch.long, device=x_text_feats.device).unsqueeze(0)
        # print(x_text_feats.shape, pos.shape)
        x = self.transformer.wte(x_text_feats) + self.transformer.wpe(pos)
        # print(x.shape)
        
        for block in self.transformer.h:
            x = block(x, x_img_feats)
        # print(x.shape)
        x = self.lm_head(self.transformer.ln_f(x))
        return x

class ImageCaptionTransformer(nn.Module):
    def __init__(self, decoder, encoder, encoder_out_dim = 1024):
        super().__init__()
        self.encoder = encoder
        self.transition_fc = nn.Linear(encoder_out_dim, 768)
        self.decoder = decoder
        self.criterion = nn.CrossEntropyLoss()
        self.forbidden_token = [
            255, 3907, 8836, 16253, 18433, 20804, 22020, 25084, 27764, 29690, 29826, 34633, 36310, 39588, 40792, 41200, 48953, 49476
        ]

    def encode(self, imgs):
        img_feats = self.encoder.forward_features(imgs)
        img_feats = self.transition_fc(img_feats) # transform to 768 dimensions
        return img_feats
    
    def forward(self, imgs, caption_ids, gt_ids):
        img_feats = self.encode(imgs)
        logits = self.decoder(caption_ids, img_feats)
        logits = torch.swapaxes(logits, 1, 2)
        # pred_ids = logits.argmax(-1)
        # In forward, the built-in collate_fn will treat the gt_ids below as tensor, but we need to transpose
        # gt_ids = torch.stack(gt_ids, dim=0).t().to(img_feats.device)
        
        # calculate loss here: reshape first!
        # Here is why: CELoss expects that (N, C, ...) where C be the number of classes, N is the batch size
        loss = self.criterion(logits, gt_ids)
        # print(loss)
        return loss
    
    # auto-regressive
    def greedy(self, img):
        self.eval()
        with torch.no_grad():
            img_feats = self.encode(img)
            # print(img_feats.shape)
        current_token_id = torch.tensor([50256]).to(img.device).unsqueeze(1)
        # print(current_token_id.shape)

        for i in range(config['max_tokens']):
            with torch.no_grad():
                logits = self.decoder(current_token_id, img_feats)
            probs = F.softmax(logits, dim=-1)
            probs_order = torch.argsort(probs, dim=-1, descending=True)

            for j in range(probs_order.shape[2]):
                if probs_order[0][i][j] not in self.forbidden_token:
                    pred_id = probs_order[0][i][j]
                    break
            if pred_id == 50256:
                break
            pred_id = pred_id.unsqueeze(0).unsqueeze(0)
            current_token_id = torch.concat((current_token_id, pred_id), dim = -1)
        
        pred_ids = current_token_id[0].cpu().tolist()[1:]
        # print('len', len(pred_ids))
        return pred_ids

    def register_hook(self, features = []):
        def hook_fn(module, ins, outs):
            features.append(outs[1].detach().cpu())
        
        handle = self.decoder.transformer.h[-1].cross_attn.register_forward_hook(hook_fn)
        return [handle]

def fixed_init(random_seed):
    myseed = random_seed  # set a random seed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(myseed)
    random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed) 
        torch.cuda.manual_seed(myseed)

class ImageDataset(Dataset): # for inference
    def __init__(self, img_dir, tfm):
        super().__init__()
        self.imgs = [os.path.join(img_dir, x) for x in os.listdir(img_dir) if x.endswith('jpg')]
        self.fnames = [x for x in os.listdir(img_dir) if x.endswith('jpg')]
        self.tfm = tfm
    
    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert('RGB')
        img = self.tfm(img)
        fname = self.fnames[idx].replace('.jpg', '')
        return img, fname
    
    def __len__(self):
        return len(self.imgs)

def inference(model, model_path, test_loader, tokenizer, json_output_path, device):
    model = model.to(device)
    state_dict = torch.load(model_path)
    print(sum([p.numel() for n, p in state_dict.items()]))
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    preds_img_sentence = {}
    with torch.no_grad():
        for batch in tqdm(test_loader): # set batch size = 1
            output_ids = model.greedy(batch[0].to(device))
            output_sentence = tokenizer.decode(output_ids)
            preds_img_sentence[str(batch[1][0])] = output_sentence
    
    with open(json_output_path, 'w') as json_f:
        json.dump(preds_img_sentence, json_f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str) #test images
    parser.add_argument("-o", type=str) #json file
    parser.add_argument("-d", type=str) #decoder weights
    args = parser.parse_args()

    test_imgs_path = args.i
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
        encoder_file = './encoder.json',
        vocab_file = './vocab.bpe'
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

    inference(model, './P2_model.ckpt', val_loader, tokenizer, output_path, device)