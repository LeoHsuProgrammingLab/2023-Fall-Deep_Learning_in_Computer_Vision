import torch
from utils import fixed_init
from dataset import Semantic_Dataset
from training import trainer
from torch.utils.data import DataLoader 
from data_aug import *
from config import config
from models import *
from testing import tester
from mean_iou_evaluate import mean_iou_score, read_masks

def main():
    # Set the random seed for reproducible experiments
    fixed_init(config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the dataset and dataloader
    train_dataset = Semantic_Dataset(path = "../../hw1/hw1_data/p3_data/train", tfm = train_tfm)
    val_dataset = Semantic_Dataset(path = "../../hw1/hw1_data/p3_data/validation", tfm = test_tfm)
    train_loader = DataLoader(train_dataset, batch_size = config['batch_size'], shuffle = True, num_workers = 2, worker_init_fn=fixed_init(config['seed']))
    val_loader = DataLoader(val_dataset, batch_size = config['batch_size'], shuffle = False, num_workers = 2, worker_init_fn=fixed_init(config['seed']))

    # Create the model
    deeplabv3 = deeplabv3_resnet50(weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1, weights_backbone = ResNet50_Weights.DEFAULT)
    deeplabv3.aux_classifier[4] = nn.Conv2d(256, 7, kernel_size = 1, stride = 1)
    deeplabv3.classifier[4] = nn.Conv2d(256, 7, kernel_size = 1, stride = 1)
    vgg16_fcn32 = model_A()
    
    from_scratch = 1 if config['model_name'] == 'model_A' else 0
    model = vgg16_fcn32 if from_scratch  else deeplabv3

    testOnly = 1
    if not testOnly:
        trainer(model, train_loader, val_loader, device, from_scratch = from_scratch)

    model_path = f'/home/leohsu-cs/DLCV2023/dlcv-fall-2023-hw1-LeoHsuProgrammingLab/p3_code/best_models/{config["model_name"]}_Epoch7.ckpt'
    tester(model, model_path, val_loader, device, output_dir = './figure/all', from_scratch = from_scratch)
    mean_iou_score(read_masks('../../hw1/hw1_data/p3_data/validation'), read_masks('./figure/all'))

if __name__ == "__main__":
    main()


