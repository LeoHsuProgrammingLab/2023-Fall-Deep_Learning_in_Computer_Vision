import torch
from tqdm.auto import tqdm

def tester(model, model_path, test_loader, device):
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))

    model.eval()
    test_pred = []
    test_gt = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            imgs, label = batch[0].to(device), batch[1]
            logits = model(imgs)
            pred = logits.argmax(dim=-1).cpu().numpy()
            test_pred.extend(pred) # 將每個batch的預測結果存起來
            test_gt.extend(label) # 將每個batch的label存起來
            
    score = 0        
    for i in range(len(test_pred)):
        if test_pred[i] == test_gt[i]:
            score += 1
    acc = score/len(test_pred)
    print("Testing Accuracy: ", acc)

    return test_pred, test_gt, acc