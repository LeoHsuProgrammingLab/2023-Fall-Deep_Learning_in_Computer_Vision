import torch
import json
import wandb
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from config import config
from dataset import ImageCaptionDataset, ImageDataset
from tokenizer import BPETokenizer
from model import *
from utils import fixed_init
from p2_evaluate import getScore

def train(model, train_loader, val_loader, tokenizer, json_output_path, device):
    model = model.to(device)
    for params in model.encoder.parameters():
        params.requires_grad = False
    for params in model.decoder.parameters():
        params.requires_grad = False
    for block in model.decoder.transformer.h:
        for params in block.cross_attn.parameters():
            params.requires_grad = True
        if block.with_adapter:
            for params in block.adapter.parameters():
                params.requires_grad = True
        if block.with_PT:
            for k, params in block.PT.named_parameters():
                # print(k, params.shape)
                params.requires_grad = True

    print(f"## Model #param={sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr = config['lr'], weight_decay=config['weight_decay'])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_CLIPscore = 0
    best_CIDEr = 0

    # wandb
    ########################################
    wandb_ = 1
    if wandb_:
        wandb.init(project = 'dlcv_hw3_p2', name = config['setting'])
        wandb.config.update({
            "epochs": config['n_epochs'],
            "batch_size": config['batch_size'],
            "learning_rate": config['lr'],
            "weight_decay": config['weight_decay'],
            "patience": config['patience'],
        })
    ########################################

    for epoch in range(config['n_epochs']):
        model.train()
        
        train_loss_list = []
        for batch in tqdm(train_loader):
            # break
            optimizer.zero_grad()
            imgs = batch['img'].to(device)
            input_cap_ids = batch['input_cap_ids'].to(device)
            gt_cap_ids = batch['gt_cap_ids'].to(device)
            loss = model(imgs, input_cap_ids, gt_cap_ids)
            loss.backward()
            train_loss_list.append(loss.item())
            optimizer.step()
        
        train_loss = sum(train_loss_list) / len(train_loss_list) if len(train_loss_list) != 0 else 10e5
        lr_scheduler.step()

        # Print the information.
        print(f"[ Train | {epoch + 1:03d}/{config['n_epochs']:03d} ] loss = {train_loss:.3f}")

        # validation
        model.eval()

        preds_img_sentence = {}
        with torch.no_grad():
            for batch in tqdm(val_loader): # set batch size = 1
                # print(batch[0])
                output_ids = model.greedy(batch[0].to(device))
                output_sentence = tokenizer.decode(output_ids)
                # print(output_sentence)
                preds_img_sentence[str(batch[1][0])] = output_sentence
        
        with open(json_output_path, 'w') as json_f:
            json.dump(preds_img_sentence, json_f)
        
        cider_score, clip_score = getScore(
            json_output_path, 
            '/home/leohsu-cs/DLCV2023/dlcv-fall-2023-hw3-LeoHsuProgrammingLab/hw3_data/p2_data/val.json', 
            '/home/leohsu-cs/DLCV2023/dlcv-fall-2023-hw3-LeoHsuProgrammingLab/hw3_data/p2_data/images/val'
        )
        # Print the information.
        print(f"[ Valid | {epoch + 1:03d}/{config['n_epochs']:03d} ] CIDEr = {cider_score:.3f}, CLIP = {clip_score:.3f}")
        if wandb_:
            wandb.log({"train_loss": train_loss, "CIDEr": cider_score, "CLIP": clip_score})

        # Save model & Early stopping
        # https://towardsdatascience.com/everything-you-need-to-know-about-saving-weights-in-pytorch-572651f3f8de
        target_state_dict = {n: p for n, p in model.named_parameters() if p.requires_grad}
        print("I saved the parameters!", len([k for k, p in target_state_dict.items()]))

        torch.save(target_state_dict, f"./ckpt/1best_{config['setting']}_e{epoch}.ckpt") # only save best to prevent output memory exceed error

        if cider_score > best_CIDEr:
            print(f"[ Valid | {epoch + 1:03d}/{config['n_epochs']:03d} ] CIDEr = {cider_score:.3f} -> best CIDEr")
            best_CIDEr = cider_score
        else:
            print(f"[ Valid | {epoch + 1:03d}/{config['n_epochs']:03d} ] CIDEr = {cider_score:.3f}")

        if clip_score > best_CLIPscore:
            print(f"[ Valid | {epoch + 1:03d}/{config['n_epochs']:03d} ] CLIP = {clip_score:.3f} -> best CLIP")
            best_CLIPscore = clip_score
        else:
            print(f"[ Valid | {epoch + 1:03d}/{config['n_epochs']:03d} ] CLIP = {clip_score:.3f}")
        
    if wandb_:
        wandb.finish()

if __name__=="__main__":
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

    cfg = Config(checkpoint='/home/leohsu-cs/DLCV2023/dlcv-fall-2023-hw3-LeoHsuProgrammingLab/hw3_data/p2_data/decoder_model.bin')
    decoder = Decoder(cfg)

    tokenizer = BPETokenizer(
        encoder_file = '/home/leohsu-cs/DLCV2023/dlcv-fall-2023-hw3-LeoHsuProgrammingLab/encoder.json',
        vocab_file = '/home/leohsu-cs/DLCV2023/dlcv-fall-2023-hw3-LeoHsuProgrammingLab/vocab.bpe'
    )

    model = ImageCaptionTransformer(
        decoder = decoder, 
        encoder = encoder
    )

    # for params in model.encoder.parameters():
    #     params.requires_grad = False
    # for params in model.decoder.parameters():
    #     params.requires_grad = False
    # for block in model.decoder.transformer.h:
    #     for params in block.cross_attn.parameters():
    #         params.requires_grad = True
    #     if block.with_adapter:
    #         for params in block.adapter.parameters():
    #             params.requires_grad = True
    # print(sum([p.numel() for n, p in model.state_dict().items()]))
    # print(f"## Model #param={sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6}M")
    # target_state_dict = {n: p for n, p in model.named_parameters() if p.requires_grad}
    # print(len([k for k, p in target_state_dict.items()]))
    # print(id(model.state_dict().items()[1]))
    # print(id(model.parameters()))
    # print(id(model.named_parameters()))
    # for n, p in model.named_parameters():
        # print(isinstance(p, nn.Module), isinstance(p, torch.Tensor))
        # print(n, p.requires_grad)
        # print(key, param.requires_grad)
    
    tfm = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor()
    ])

    train_set = ImageCaptionDataset(
        img_dir='/home/leohsu-cs/DLCV2023/dlcv-fall-2023-hw3-LeoHsuProgrammingLab/hw3_data/p2_data/images/train',
        json_file='/home/leohsu-cs/DLCV2023/dlcv-fall-2023-hw3-LeoHsuProgrammingLab/hw3_data/p2_data/train.json',
        tokenizer=tokenizer, 
        tfm=encode_transform
    )

    val_set = ImageDataset(
        img_dir='/home/leohsu-cs/DLCV2023/dlcv-fall-2023-hw3-LeoHsuProgrammingLab/hw3_data/p2_data/images/val', 
        tfm=encode_transform
    )

    train_loader = DataLoader(
        train_set, batch_size=config['batch_size'], 
        shuffle=True, num_workers=3, worker_init_fn=fixed_init(666),
        collate_fn=train_set.collate_fn
    )

    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=True, num_workers=3, worker_init_fn=fixed_init(666)
    )

    train(model, train_loader, val_loader, tokenizer, '/home/leohsu-cs/DLCV2023/dlcv-fall-2023-hw3-LeoHsuProgrammingLab/p2_code/output/preds.json', device)