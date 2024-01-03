import torch
from tqdm.auto import tqdm
from mean_iou_evaluate import mean_iou_score
from plot import pred2image

def tester(model, model_path, test_loader, device, output_dir, from_scratch = False):
    pred_mask_list = []
    label_mask_list = []
    
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    
    target_figure = ['0013_sat.jpg', '0062_sat.jpg', '0104_sat.jpg']

    with torch.no_grad():
        for batch in tqdm(test_loader):
            imgs, labels, file_name = batch
            imgs, labels = imgs.to(device), labels.squeeze(1).to(device, dtype = torch.long)
            if from_scratch:
                logits = model(imgs)
            else:
                logits = model(imgs)['out']
            pred_masks = logits.argmax(dim = 1).cpu()

            for i in range(len(file_name)):
                pred2image(pred_masks[i], file_name[i], output_dir)