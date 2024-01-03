import torch
from byol_pytorch import BYOL
from torchvision import models
from config import config
from dataset import P2_mini
from torch.utils.data import DataLoader
from data_aug import pretext_tfm
from utils import fixed_init
from tqdm.auto import tqdm

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

if __name__ == '__main__':
    pretrain()