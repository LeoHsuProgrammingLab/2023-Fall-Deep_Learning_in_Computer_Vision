import torch
import numpy as np
import torchvision.models as models
from torch.utils.data import DataLoader
from data_aug import test_tfm
from dataset import DigitsDataset
from config import config
from utils import fixed_init
from model import MyResnet50
from sklearn.manifold import TSNE

'''
Refernce:
1. https://kozodoi.me/blog/20210527/extracting-features
'''

features = {}
def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook
# model.layer4.register_forward_hook(get_features('l4'))

def inference(model, test_loader, device):
    model = model.to(device)
    model.eval()
    pred_list = []
    gt_list = []

    all_feats, all_labels = None
    with torch.no_grad():
        for batch in test_loader:
            imgs, labels = batch[0].to(device), batch[1].to(device)
            logits, l4_features = model(imgs)
            all_feats = l4_features if all_feats is None else torch.cat((all_feats, l4_features), dim=0)  
            print(all_feats.shape) 
            pred_list.extend(logits.argmax(dim=-1).tolist())
            gt_list.extend(labels.tolist())

    tsne = TSNE(n_components=2, random_state=666)

    
    print(round(eval(gt_list, pred_list), 3))

def eval(gt_list, pred_list):
    gt = np.array(gt_list)
    pred = np.array(pred_list)
    acc = np.mean(gt == pred)
    return acc

if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MyResnet50()
    print(model)

    settings = ['lower_bound', 'DANN', 'upper_bound']
    setting = settings[1]
    domains = ['mnistm', 'svhn', 'usps']
    domain = domains[1]
    model.load_state_dict(torch.load(f"./best_models/{setting}/{domain}_best.ckpt"))

    tgt_domain = domains[1]
    img_path = f'/home/leohsu-cs/DLCV2023/hw2-LeoHsuProgrammingLab/hw2_data/digits/{tgt_domain}/data/'
    
    test_csv = f'/home/leohsu-cs/DLCV2023/hw2-LeoHsuProgrammingLab/hw2_data/digits/{tgt_domain}/val.csv'
    test_set = DigitsDataset(path = img_path, transform = test_tfm, csv = test_csv)
    test_loader = DataLoader(test_set, batch_size = config['batch_size'], shuffle=False, num_workers=3, worker_init_fn=fixed_init(666))

    inference(model, test_loader, device)
    