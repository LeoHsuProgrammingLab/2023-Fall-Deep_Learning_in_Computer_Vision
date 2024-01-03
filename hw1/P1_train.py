import torch.nn as nn
import torch
from tqdm.auto import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import wandb
import matplotlib.pyplot as plt

def trainer(train_loader, val_loader, model, device, model_type = "B", config = None):
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Can try SAM, GAM: https://github.com/davda54/sam#readme
    optimizer = torch.optim.AdamW(model.parameters(), lr = config['lr'], weight_decay = config['weight_decay'])
    lr_scheduler_B = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    lr_scheduler_A = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 10, eta_min = config['lr']/20)
    lr_scheduler = lr_scheduler_B if model_type == "B" else lr_scheduler_A

    best_acc = 0
    stale = 0

    train_loss_list = []
    train_acc_list = []

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="dlcv_hw1_p1",
        name = config['model_name'] + "",
        
        # track hyperparameters and run metadata
        config={
            "learning_rate": config['lr'],
            "epochs": config['num_epochs'],
            "patience": config['patience'], 
            "batch_size": config['batch_size'],
            "model_name": config['model_name'],
            "weight_decay": config['weight_decay'],
            "log": config['log']
        }
    )

    for epoch in range(config['num_epochs']):
        # Training
        model.train()
        train_loss = 0
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
            # variable of plots
            all_latent, all_labels = None, None
            for batch in tqdm(val_loader):
                imgs, labels = batch[0].to(device), batch[1].to(device)
                logits = model(imgs)
                # Get latent from backbone
                if model_type == "A":
                    backbone_latent = model.get_backbone_latent(imgs).cpu().detach().numpy()
                    all_latent = backbone_latent if all_latent is None else np.vstack((all_latent, backbone_latent))
                    all_labels = labels.cpu().detach().numpy() if all_labels is None else np.concatenate((all_labels, labels.cpu().detach().numpy()))

                loss = criterion(logits, labels)
                acc = (logits.argmax(dim=-1) == labels).float().mean()
                valid_loss_list.append(loss.item())
                valid_acc_list.append(acc)

            if model_type == "A":
                # Plot the latent space
                all_latent = all_latent.reshape(all_latent.shape[0], -1)
                # PCA
                pca = PCA(n_components=2)
                dim_x = pca.fit_transform(all_latent)
                plt.figure()
                plt.title("PCA figure in epoch {}".format(epoch))
                plt.scatter(dim_x[:, 0], dim_x[:, 1], c=all_labels)
                plt.savefig("./figure/PCA_{}.png".format(epoch))
                # t-SNE
                tsne = TSNE(n_components=2)
                dim_x = tsne.fit_transform(all_latent)
                plt.figure()
                plt.title("t-SNE figure in epoch {}".format(epoch))
                plt.scatter(dim_x[:, 0], dim_x[:, 1], c=all_labels)
                plt.savefig("./figure/t-SNE_{}.png".format(epoch))

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
            torch.save(model.state_dict(), f"./best_models/{config['model_name']}_best.ckpt") # only save best to prevent output memory exceed error
            best_acc = valid_acc
            stale = 0
        else:
            stale += 1
            if stale >= config['patience']:
                print(f"No improvment {config['patience']} consecutive epochs, early stopping")
                break

    wandb.finish()