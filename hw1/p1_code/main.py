from dataset import P1_Dataset, train_valid_split
from training import trainer
from testing import tester
from data_aug import train_tfm_A, test_tfm_A, train_tfm_B, test_tfm_B
from config import config_A, config_B
from torch.utils.data import DataLoader
import torch
from models import get_model_list 
from utils import fixed_init

def main():
    fixed_init(666)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(torch.version.cuda)
    # print(torch.cuda.get_arch_list())

    # model
    model_type = "A"
    model_list = get_model_list()
    # init depends on model_type
    cur_model = None
    config = None
    train_tfm, test_tfm = None, None
    if model_type == "A":
        config = config_A 
        train_tfm = train_tfm_A
        test_tfm = test_tfm_A
        cur_model = model_list[0]
    else:
        config = config_B
        train_tfm = train_tfm_B
        test_tfm = test_tfm_B
        cur_model = model_list[1]

    # current target
    testOnly = 1

    if not testOnly:
        # train dataloader
        train_dataset = P1_Dataset("../hw1_data/p1_data/train_50", train_tfm)
        train_set, valid_set = train_valid_split(train_dataset, 0.05, 0)
        # ref: https://pytorch.org/docs/stable/notes/randomness.html#reproducibility
        train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=2, worker_init_fn=fixed_init(666))
        val_loader = DataLoader(valid_set, batch_size=config['batch_size'], shuffle=True, num_workers=2, worker_init_fn=fixed_init(666))
        # train
        trainer(train_loader, val_loader, cur_model, device, model_type = model_type, config = config) 

    test_set = P1_Dataset("../hw1_data/p1_data/val_50", test_tfm)
    test_loader = DataLoader(test_set, batch_size = 32, shuffle = False, num_workers=2)
    
    # test
    model_path = f"./best_models/{config['model_name']}.ckpt"
    tester(cur_model, model_path, test_loader, device)

if __name__ == '__main__':
    main()