import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from config import config
from dataset import DigitsDataset
from data_aug import train_tfm, test_tfm
from utils import fixed_init
from model import DomainClassifier, MyResnet50


def trainer(model, device, train_loader, val_loader, setting, domain):
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = config['lr'], weight_decay = config['weight_decay'])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)

    best_acc = 0
    stale = 0

    train_loss_list = []
    train_acc_list = []

    # wandb
    ########################################
    wandb_ = False
    if wandb_:
        wandb.init(project = 'dlcv_hw2_p3', name = config['model_name'] + '_' + setting + '_' + domain)
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
        for batch in train_loader:
            imgs, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            logits, l4_features = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            acc = (logits.argmax(dim=-1) == labels).float().mean()
            train_loss_list.append(loss.item())
            train_acc_list.append(acc.cpu().numpy())

        lr_scheduler.step()

        train_loss = np.mean(train_loss_list)
        train_acc = np.mean(train_acc_list)

        # Print the information.
        print(f"[ Train | {epoch + 1:03d}/{config['n_epochs']:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        # Validation
        model.eval()

        valid_loss_list = []
        valid_acc_list = []

        with torch.no_grad():
            for batch in tqdm(val_loader):
                imgs, labels = batch[0].to(device), batch[1].to(device)
                logits, l4_features = model(imgs)

                loss = criterion(logits, labels)
                acc = (logits.argmax(dim=-1) == labels).float().mean()
                valid_loss_list.append(loss.item())
                valid_acc_list.append(acc.cpu().numpy())
        
        valid_loss = np.mean(valid_loss_list)
        valid_acc = np.mean(valid_acc_list)

        # Print the information.
        print(f"[ Valid | {epoch + 1:03d}/{config['n_epochs']:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
        if wandb_:
            wandb.log({"train_loss": train_loss, "val_loss": valid_loss, "train_acc": train_acc, "val_acc": valid_acc})
        
        # Save model & Early stopping
        if valid_acc > best_acc:
            print(f"[ Valid | {epoch + 1:03d}/{config['n_epochs']:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
        else:
            print(f"[ Valid | {epoch + 1:03d}/{config['n_epochs']:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        if valid_acc > best_acc:
            print(f"Best model found at epoch {epoch}, saving model")
            torch.save(model.state_dict(), f"./best_models/{setting}/{domain}_best.ckpt") # only save best to prevent output memory exceed error
            best_acc = valid_acc
            stale = 0
        else:
            stale += 1
            if stale >= config['patience']:
                print(f"No improvment {config['patience']} consecutive epochs, early stopping")
                break
    if wandb_:
        wandb.finish()

def DANN_trainer(
        model, 
        domain_classifier, 
        device, 
        src_train_loader, 
        tgt_train_loader, 
        tgt_valid_loader, 
        domain,
        setting
    ):

        model = model.to(device)
        domain_classifier = domain_classifier.to(device)

        label_criterion = nn.CrossEntropyLoss()
        domain_criterion = nn.BCEWithLogitsLoss()
        label_optimizer = torch.optim.SGD(model.parameters(), lr = config['lr'], weight_decay=config['weight_decay'])
        domain_optimizer = torch.optim.SGD(domain_classifier.parameters(), lr = config['lr'], weight_decay = config['weight_decay'])

        lr_scheduler1 = torch.optim.lr_scheduler.StepLR(label_optimizer, step_size=20, gamma=0.3)
        lr_scheduler2 = torch.optim.lr_scheduler.StepLR(domain_optimizer, step_size=20, gamma=0.3)
        # lr_scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(label_optimizer, T_max=10, eta_min=1e-10)
        # lr_scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(domain_optimizer, T_max=10)

        best_acc = 0
        stale = 0

        gamma = 10

        train_loss_list = []
        train_acc_list = []
        valid_loss_list = []
        valid_acc_list = []

        # wandb
        ########################################
        wandb_ = True
        if wandb_:
            wandb.init(project = 'dlcv_hw2_p3', name = config['model_name'] + '_' + setting + '_' + domain)
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
            # hyper-parameters 
            p = epoch / config['n_epochs']
            lambda_p = (2 / (1 + np.exp(gamma * -1 * p))) - 1 
            print('lambda_p: ', lambda_p)

            for src_batch, tgt_batch in tqdm(zip(src_train_loader, tgt_train_loader)):
                label_optimizer.zero_grad()
                domain_optimizer.zero_grad()

                src_imgs, src_labels = src_batch[0].to(device), src_batch[1].to(device)
                tgt_imgs = tgt_batch[0].to(device)
                mixed_imgs = torch.cat([src_imgs, tgt_imgs], dim = 0)

                # print(src_imgs.shape[0], tgt_imgs.shape[0])
                domain_labels = torch.zeros(src_imgs.shape[0] + tgt_imgs.shape[0]).to(device)
                domain_labels[src_imgs.shape[0]:] = 1
                
                src_logits, _ = model(src_imgs)
                label_loss = label_criterion(src_logits, src_labels)
                acc = (src_logits.argmax(dim=-1) == src_labels).float().mean()

                _, l4_features = model(mixed_imgs)
                domain_logits = domain_classifier(l4_features).squeeze()
                domain_loss = domain_criterion(domain_logits, domain_labels)

                loss = label_loss + domain_loss * lambda_p
                loss.backward()
                label_optimizer.step()
                domain_optimizer.step()

                train_loss_list.append(loss.item())
                train_acc_list.append(acc.cpu().numpy())

            train_loss = np.mean(train_loss_list)
            train_acc = np.mean(train_acc_list)
            lr_scheduler1.step()
            lr_scheduler2.step()

            # Print the information.
            print(f"[ Train | {epoch + 1:03d}/{config['n_epochs']:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

            model.eval()
            with torch.no_grad():
                for batch in tqdm(tgt_valid_loader):
                    tgt_imgs, tgt_labels = batch[0].to(device), batch[1].to(device)
                    tgt_logits, tgt_l4_feats = model(tgt_imgs)
                    loss = label_criterion(tgt_logits, tgt_labels)
                    acc = (tgt_logits.argmax(dim=-1) == tgt_labels).float().mean()

                    valid_loss_list.append(loss.item())
                    valid_acc_list.append(acc.cpu().numpy())

            valid_loss = np.mean(valid_loss_list)
            valid_acc = np.mean(valid_acc_list)

            # Print the information.
            print(f"[ Valid | {epoch + 1:03d}/{config['n_epochs']:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
            if wandb_:
                wandb.log({"train_loss": train_loss, "val_loss": valid_loss, "train_acc": train_acc, "val_acc": valid_acc})
            
            # Save model & Early stopping
            if valid_acc > best_acc:
                print(f"[ Valid | {epoch + 1:03d}/{config['n_epochs']:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
            else:
                print(f"[ Valid | {epoch + 1:03d}/{config['n_epochs']:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

            if valid_acc > best_acc:
                print(f"Best model found at epoch {epoch}, saving model")
                torch.save(model.state_dict(), f"./best_models/{setting}/{domain}_best.ckpt") # only save best to prevent output memory exceed error
                best_acc = valid_acc
                stale = 0
            else:
                stale += 1
                if stale >= config['patience']:
                    print(f"No improvment {config['patience']} consecutive epochs, early stopping")
                    break
        if wandb_:
            wandb.finish()

if __name__=='__main__':
    fixed_init(666)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyResnet50()

    settings = ['lower_bound', 'DANN', 'upper_bound']
    setting = settings[1]
    domains = ['mnistm', 'svhn', 'usps']

    if setting == settings[1]:
        domain_clf = DomainClassifier()

        src = domains[0]
        tgt = domains[2]

        src_img_path = f'/home/leohsu-cs/DLCV2023/hw2-LeoHsuProgrammingLab/hw2_data/digits/{src}/data/'
        tgt_img_path = f'/home/leohsu-cs/DLCV2023/hw2-LeoHsuProgrammingLab/hw2_data/digits/{tgt}/data/'

        src_train_csv = f'/home/leohsu-cs/DLCV2023/hw2-LeoHsuProgrammingLab/hw2_data/digits/{src}/train.csv'
        tgt_train_csv = f'/home/leohsu-cs/DLCV2023/hw2-LeoHsuProgrammingLab/hw2_data/digits/{tgt}/train.csv'
        tgt_val_csv = f'/home/leohsu-cs/DLCV2023/hw2-LeoHsuProgrammingLab/hw2_data/digits/{tgt}/val.csv'

        src_train_set = DigitsDataset(path = src_img_path, transform = train_tfm, csv = src_train_csv)
        tgt_train_set = DigitsDataset(path = tgt_img_path, transform = train_tfm, csv = tgt_train_csv)
        tgt_val_set = DigitsDataset(path = tgt_img_path, transform = test_tfm, csv = tgt_val_csv)

        src_train_loader = DataLoader(src_train_set, batch_size = config['batch_size'], shuffle=True, num_workers=4, worker_init_fn=fixed_init(666))
        tgt_train_loader = DataLoader(tgt_train_set, batch_size = config['batch_size'], shuffle=True, num_workers=4, worker_init_fn=fixed_init(666))
        tgt_val_loader = DataLoader(tgt_val_set, batch_size = config['batch_size'], shuffle=False, num_workers=4, worker_init_fn=fixed_init(666))
        DANN_trainer(model, domain_clf, device, src_train_loader, tgt_train_loader, tgt_val_loader, tgt, setting)
    else:
        domain = domains[2]
        img_path = f'/home/leohsu-cs/DLCV2023/hw2-LeoHsuProgrammingLab/hw2_data/digits/{domain}/data/'
        
        train_csv = f'/home/leohsu-cs/DLCV2023/hw2-LeoHsuProgrammingLab/hw2_data/digits/{domain}/train.csv'
        # test_csv = f'/home/leohsu-cs/DLCV2023/hw2-LeoHsuProgrammingLab/hw2_data/digits/{domain}/test.csv'
        val_csv = f'/home/leohsu-cs/DLCV2023/hw2-LeoHsuProgrammingLab/hw2_data/digits/{domain}/val.csv'

        train_set = DigitsDataset(path = img_path, transform = train_tfm, csv = train_csv)
        val_set = DigitsDataset(path = img_path, transform = test_tfm, csv = val_csv)
        # test_set = DigitsDataset(path = img_path, transform = test_tfm, csv = test_csv)

        train_loader = DataLoader(train_set, batch_size = config['batch_size'], shuffle=True, num_workers=4, worker_init_fn=fixed_init(666))
        val_loader = DataLoader(val_set, batch_size = config['batch_size'], shuffle=True, num_workers=4, worker_init_fn=fixed_init(666))
        # test_loader = DataLoader(test_set, batch_size = config['batch_size'], shuffle=False, num_workers=4, worker_init_fn=fixed_init(666))
    
        trainer(model, device, train_loader, val_loader, setting, domain)

