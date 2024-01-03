import torch.nn as nn
import torch
import wandb
import os
import numpy as np
import random
from tqdm.auto import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from byol_pytorch import BYOL

class P2_mini(Dataset):
    def __init__(self, path, tfm = None, file_names = None):
        super(P2_mini).__init__()
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
            
        return im
    
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
            
        return im, label
    
pretext_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std  = [0.229, 0.224, 0.225]),
])

train_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(p = 0.2),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std  = [0.229, 0.224, 0.225]),
])

config = {
    'lr': 0.0003,
    'batch_size': 64,
    "weight_decay": 0.0005,
    'patience': 50,
    'num_epochs': 200,
    'setting': 'C',
    
    'pretext_lr': 0.0001,
    'pretext_batch_size': 128,
    'pretext_epochs': 1000,
}

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

def wandb_init():
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="dlcv_hw1_p2",
        name = 'model_' + config['setting'],
        
        # track hyperparameters and run metadata
        config={
            "learning_rate": config['lr'],
            "epochs": config['num_epochs'],
            "patience": config['patience'], 
            "batch_size": config['batch_size'],
            "weight_decay": config['weight_decay'],
            "log": ""
        }
    )

def fine_tuner(model, train_loader, val_loader, device):
    
    criterion = nn.CrossEntropyLoss()
    # Can try SAM, GAM: https://github.com/davda54/sam#readme
    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], weight_decay = config['weight_decay'])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 10, eta_min = config['lr']/20)

    best_acc = 0
    stale = 0

    model = model.to(device)
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

def pretrain():
    fixed_init(2023)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = models.resnet50(weights = None)

    learner = BYOL(
        resnet,
        image_size = 128,
        hidden_layer = 'avgpool',
        use_momentum = False # turn on momentum
    )
    learner = learner.to(device)

    optimizer = torch.optim.AdamW(learner.parameters(), lr=config['pretext_lr'], weight_decay=config['weight_decay'])
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 10, eta_min = config['lr']/20)
    dataset = P2_mini("../hw1_data/p2_data/mini/train", pretext_tfm)
    train_loader = DataLoader(dataset, batch_size = config['pretext_batch_size'], shuffle = True, num_workers=8, worker_init_fn=fixed_init(2023))

    for i in range(config['pretext_epochs']):
        print(f"Epoch {i}")
        for imgs_batch in tqdm(train_loader):
            imgs_batch = imgs_batch.to(device)
            loss = learner(imgs_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        torch.save(resnet.state_dict(), './pretext_models/resnet50.ckpt')