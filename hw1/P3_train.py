import torch.nn as nn
import torch
import numpy as np
from config import config
from tqdm.auto import tqdm
import wandb
import os
import imageio


def wandb_init():
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="dlcv_hw1_p3",
        name = config['model_name'] + "_",
        
        # track hyperparameters and run metadata
        config={
            "learning_rate": config['lr'],
            "epochs": config['num_epochs'],
            "patience": config['patience'], 
            "batch_size": config['batch_size'],
            "model_name": config['model_name'],
            "weight_decay": config['weight_decay'],
            "log": ""
        }
    )

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

def mean_iou_score(pred, labels):
    '''
    Compute mean IoU score over 6 classes
    '''
    mean_iou = 0
    for i in range(6):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou / 6
        print('class #%d : %1.5f'%(i, iou))
    print('\nmean_iou: %f\n' % mean_iou)

    return mean_iou

def trainer(model, train_loader, val_loader, device, from_scratch = False):
    criterion = nn.CrossEntropyLoss(ignore_index = 6)
    optimizer = torch.optim.SGD(model.parameters(), lr = config['lr'], weight_decay = config['weight_decay'], momentum=0.95)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.5)

    best_miou = 0
    stale = 0

    model = model.to(device)
    wandb_init()

    target_epoch = [0, 7, 14]

    for epoch in range(config['num_epochs']):
        train_loss_list = []

        model.train()
        for batch in tqdm(train_loader):
            imgs, labels = batch[0].to(device), batch[1].squeeze(1).to(device, dtype = torch.long)
            optimizer.zero_grad()
            if from_scratch:
                logits = model(imgs)
            else:
                logits = model(imgs)['out']
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss_list.append(loss.item())
        lr_scheduler.step()

        train_loss = sum(train_loss_list) / len(train_loss_list)
        print(f"[ Train | {epoch + 1:03d}/{config['num_epochs']:03d} ] loss = {train_loss:.5f}")

        # Validation
        val_loss_list = []
        pred_mask_list = []
        label_mask_list = []

        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_loader):
                imgs, labels = batch[0].to(device), batch[1].squeeze(1).to(device, dtype = torch.long)
                if from_scratch:
                    logits = model(imgs)
                else:
                    logits = model(imgs)['out']
                loss = criterion(logits, labels)
                val_loss_list.append(loss.item())

                pred_masks = logits.argmax(dim = 1)
                pred_mask_list.append(pred_masks.detach().cpu())
                label_mask_list.append(labels.detach().cpu())
        
        pred_masks_result = torch.cat(pred_mask_list).numpy()
        label_masks_result = torch.cat(label_mask_list).numpy()
        val_loss = sum(val_loss_list) / len(val_loss_list)
        val_miou = mean_iou_score(pred_masks_result, label_masks_result)
        
        if epoch in target_epoch:
            torch.save(model.state_dict(), f'./best_models/{config["model_name"]}_Epoch{epoch}.ckpt')

        # Save model & Early stopping
        if val_miou > best_miou:
            print(f"[ Valid | {epoch + 1:03d}/{config['num_epochs']:03d} ] loss = {val_loss:.5f}, miou = {val_miou:.5f} -> best")
            print(f"Best model found at epoch {epoch}, saving model")
            best_miou = val_miou
            torch.save(model.state_dict(), f'./best_models/{config["model_name"]}.ckpt')
            stale = 0
        else:
            print(f"[ Valid | {epoch + 1:03d}/{config['num_epochs']:03d} ] loss = {val_loss:.5f}, miou = {val_miou:.5f}")
            stale += 1
            if stale >= config['patience']:
                print(f"No improvment {config['patience']} consecutive epochs, early stopping")
                break
        print('\n')
        
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "val_miou": val_miou})
    
    wandb.finish()

