import torch
from fine_tune import fine_tuner
from models import MyResnet50
from dataset import P2_office
from torch.utils.data import DataLoader
from data_aug import train_tfm, test_tfm
from utils import fixed_init
from config import config
from testing import tester

def main():
    fixed_init(2023)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone_list = ["./pretext_models/resnet50.ckpt", "../hw1_data/p2_data/pretrain_model_SL.pt"]
    model = MyResnet50(backbone_ckpt = backbone_list[0], setting = config['setting'])

    train_set = P2_office("../hw1_data/p2_data/office/train", train_tfm)
    val_set = P2_office("../hw1_data/p2_data/office/val", test_tfm)
    train_loader = DataLoader(train_set, batch_size = config['batch_size'], shuffle = True, num_workers=2, worker_init_fn=fixed_init(2023))
    val_loader = DataLoader(val_set, batch_size = config['batch_size'], shuffle = False, num_workers=2, worker_init_fn=fixed_init(2023))
    print(len(val_set), len(train_set))

    test_only = 1
    if not test_only:
        fine_tuner(model, train_loader, val_loader, device)
    # test
    tester(model, f"./fine_tuned_resnet50/model_{config['setting']}.ckpt", val_loader, device)

if __name__ == '__main__':
    main()
