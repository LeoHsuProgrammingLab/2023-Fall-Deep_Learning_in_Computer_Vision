import torch
import numpy as np
import matplotlib.pyplot as plt
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
2. 
'''

def plot(model, src_val_loader, tgt_val_loader, device, tgt_domain):
    model = model.to(device)
    model.eval()
    pred_list = []
    gt_list = []

    all_feats, all_labels, all_domains = None, None, None
    with torch.no_grad():
        for batch in src_val_loader:
            imgs, labels = batch[0].to(device), batch[1].to(device)
            logits, l4_features = model(imgs)
            domains = torch.zeros((imgs.shape[0], ))
            all_feats = l4_features if all_feats is None else torch.cat((all_feats, l4_features), dim=0) 
            all_labels = labels if all_labels is None else torch.cat((all_labels, labels), dim=0) 
            all_domains = torch.zeros((imgs.shape[0], )) if all_domains is None else torch.cat((all_domains, domains), dim=0)

            pred_list.extend(logits.argmax(dim=-1).tolist())
            gt_list.extend(labels.tolist())

        for batch in tgt_val_loader:
            imgs, labels = batch[0].to(device), batch[1].to(device)
            logits, l4_features = model(imgs)
            domains = torch.ones((imgs.shape[0], ))
            all_feats = torch.cat((all_feats, l4_features), dim=0) 
            all_labels = torch.cat((all_labels, labels), dim=0)
            all_domains = torch.cat((all_domains, domains), dim=0) 

            # print(all_feats.shape, all_labels.shape, all_domains.shape) 

            pred_list.extend(logits.argmax(dim=-1).tolist())
            gt_list.extend(labels.tolist())

    all_feats = all_feats.cpu().numpy()
    all_labels = all_labels.cpu().numpy()
    all_domains = all_domains.cpu().numpy()

    tsne = TSNE(n_components=2, random_state=1003)
    tsne_x = tsne.fit_transform(all_feats)
    point_size = 10
    plt.figure()
    plt.title("t-SNE for Layer4 with Classes")
    scatter = plt.scatter(tsne_x[:, 0], tsne_x[:, 1], c=all_labels, s=point_size)
    plt.legend(*scatter.legend_elements(), bbox_to_anchor = (1.135, 1))
    plt.savefig(f"./figure/tSNE/{tgt_domain}_labels.png")

    plt.title("t-SNE for Layer4 with Domains")
    scatter = plt.scatter(tsne_x[:, 0], tsne_x[:, 1], c=all_domains, s=point_size)
    # print(scatter.legend_elements())
    plt.legend(handles = scatter.legend_elements()[0], labels = ['source', 'target'], bbox_to_anchor = (1, 1))
    plt.savefig(f"./figure/tSNE/{tgt_domain}_domains.png")

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

    src_domain = domains[0]
    src_img_path = f'/home/leohsu-cs/DLCV2023/hw2-LeoHsuProgrammingLab/hw2_data/digits/{src_domain}/data/'
    src_val_csv = f'/home/leohsu-cs/DLCV2023/hw2-LeoHsuProgrammingLab/hw2_data/digits/{src_domain}/val.csv'
    src_val_set = DigitsDataset(path = src_img_path, transform = test_tfm, csv = src_val_csv)
    src_val_loader = DataLoader(src_val_set, batch_size = config['batch_size'], shuffle=False, num_workers=3, worker_init_fn=fixed_init(666))
    
    tgt_domain = domains[1]
    tgt_img_path = f'/home/leohsu-cs/DLCV2023/hw2-LeoHsuProgrammingLab/hw2_data/digits/{tgt_domain}/data/'
    tgt_val_csv = f'/home/leohsu-cs/DLCV2023/hw2-LeoHsuProgrammingLab/hw2_data/digits/{tgt_domain}/val.csv'
    tgt_val_set = DigitsDataset(path = tgt_img_path, transform = test_tfm, csv = tgt_val_csv)
    tgt_val_loader = DataLoader(tgt_val_set, batch_size = config['batch_size'], shuffle=False, num_workers=3, worker_init_fn=fixed_init(666))
    
    model.load_state_dict(torch.load(f"./best_models/{setting}/{tgt_domain}_best.ckpt"))
    plot(model, src_val_loader, tgt_val_loader, device, tgt_domain)