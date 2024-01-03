import torch
from config import config
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from data_aug import transform
from dataset import DigitsDataset
from utils import fixed_init
from model import *

def trainer(model, train_loader, test_loader, device):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'])
    guided_w = [0, 0.5, 2]
    
    for epoch in range(config['n_epochs']):
        print(f"Current Epoch: {epoch + 1:03d}/{config['n_epochs']:03d}")
        model.train()
        # lr scheduler for different epoch
        optimizer.param_groups[0]['lr'] = config['lr'] * (1 - epoch / config['n_epochs'])
        train_loss_list = []
        
        for img, label in train_loader:
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            loss = model(img, label)
            loss.backward()
            optimizer.step()

            train_loss_list.append(loss)
        # Print the information.
        print(f"[ Train | {epoch + 1:03d}/{config['n_epochs']:03d} ] loss = {sum(train_loss_list)/len(train_loss_list):.5f}")

        model.eval()
        with torch.no_grad():
            n_sample = 10 * config['n_classes']
            for w in guided_w:
                x_gen, x_target_step_list = model.sample(n_sample, (3, 28, 28), device, w)
                grid1 = make_grid(x_gen * -1 + 1, nrow=10)
                grid2 = make_grid(x_gen, nrow=10)
                save_image(grid1, f'./figure/iepoch_{epoch}_{w}.jpg')
                save_image(grid2, f'./figure/epoch_{epoch}_{w}.jpg')

        torch.save(model.state_dict(), f'./ckpt/model_e{epoch}.ckpt')

if __name__=='__main__':
    train_dataset = DigitsDataset(
        path = '/home/leohsu-cs/DLCV2023/hw2-LeoHsuProgrammingLab/hw2_data/digits/mnistm/data/', 
        transform = transform,
        csv = '/home/leohsu-cs/DLCV2023/hw2-LeoHsuProgrammingLab/hw2_data/digits/mnistm/train.csv'
    )

    test_dataset = DigitsDataset(
        path = '/home/leohsu-cs/DLCV2023/hw2-LeoHsuProgrammingLab/hw2_data/digits/mnistm/data/', 
        transform = transform,
        csv = '/home/leohsu-cs/DLCV2023/hw2-LeoHsuProgrammingLab/hw2_data/digits/mnistm/test.csv'
    )

    train_loader = DataLoader(train_dataset, batch_size = config['batch_size'], shuffle = True, num_workers=3, worker_init_fn=fixed_init(1027))
    test_loader = DataLoader(test_dataset, batch_size = config['batch_size'], shuffle = False, num_workers=3, worker_init_fn=fixed_init(1027))

    model = DDPM(
        backbone_model = UnetBackbone(in_channels=3, n_latent_dim=config['n_latent_dim'], n_classes=config['n_classes']),
        betas = [1e-4, 2e-2],
        n_T = config['n_T'],
        device = config['device'],
        drop_prb = config['drop_prb']
    )


    trainer(model, train_loader, test_loader, config['device'])