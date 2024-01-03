import torch.nn as nn
import torch
import wandb
from tqdm.auto import tqdm
from config import config
import os

def wandb_init():
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="dlcv_hw1_p2",
        name = config['model_name'] + "_" + config['setting'],
        
        # track hyperparameters and run metadata
        config={
            "learning_rate": config['lr'],
            "epochs": config['num_epochs'],
            "patience": config['patience'], 
            "batch_size": config['batch_size'],
            "weight_decay": config['weight_decay'],
            "log": config['log'],
        }
    )

def fine_tuner(model, train_loader, val_loader, device):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    # Can try SAM, GAM: https://github.com/davda54/sam#readme
    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], weight_decay = config['weight_decay'])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 10, eta_min = config['lr']/20)

    best_acc = 0
    stale = 0

    wandb_init()
     
    for epoch in range(config['num_epochs']):
        train_loss = 0
        train_loss_list = []
        train_acc_list = []
        
        # Training
        model.train()
        for batch in tqdm(train_loader):
            imgs, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            acc = (logits.argmax(dim=-1) == labels).float().mean()
            train_loss_list.append(loss.item())
            train_acc_list.append(acc)
        
        lr_scheduler.step()

        train_loss = sum(train_loss_list) / len(train_loss_list)
        train_acc = sum(train_acc_list) / len(train_acc_list)

        # Print the information.
        print(f"[ Train | {epoch + 1:03d}/{config['num_epochs']:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        # Validation
        model.eval()

        valid_loss_list = []
        valid_acc_list = []

        with torch.no_grad():
            for batch in tqdm(val_loader):
                imgs, labels = batch[0].to(device), batch[1].to(device)
                logits = model(imgs)
                loss = criterion(logits, labels)
                acc = (logits.argmax(dim=-1) == labels).float().mean()
                valid_loss_list.append(loss.item())
                valid_acc_list.append(acc)

        valid_loss = sum(valid_loss_list) / len(valid_loss_list)
        valid_acc = sum(valid_acc_list) / len(valid_acc_list)

        # Print the information.
        print(f"[ Valid | {epoch + 1:03d}/{config['num_epochs']:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
        wandb.log({"train_loss": train_loss, "val_loss": valid_loss, "train_acc": train_acc, "val_acc": valid_acc})
        
        # Save model & Early stopping
        if valid_acc > best_acc:
            print(f"[ Valid | {epoch + 1:03d}/{config['num_epochs']:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
        else:
            print(f"[ Valid | {epoch + 1:03d}/{config['num_epochs']:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        if valid_acc > best_acc:
            print(f"Best model found at epoch {epoch}, saving model")
            if not os.path.exists("./fine_tuned_models"):
                os.makedirs("./fine_tuned_models")
            torch.save(model.state_dict(), f"./fine_tuned_models/{config['model_name']}_{config['setting']}.ckpt") # only save best to prevent output memory exceed error
            best_acc = valid_acc
            stale = 0
        else:
            stale += 1
            if stale >= config['patience']:
                print(f"No improvment {config['patience']} consecutive epochs, early stopping")
                break

    wandb.finish()
