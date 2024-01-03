import pandas as pd

def evaluate_csv(pred_path):  
    output = pd.read_csv(pred_path)
    pred = output['label'].tolist()
    gt = output['filename'].tolist()
    score = 0
    for i in range(len(pred)):
        if pred[i] == int(gt[i].split("/")[-1].split("_")[0]):
            score += 1
    acc = score/len(pred)
    print("Testing Accuracy: ", acc)
    return acc

if __name__ == "__main__":
    evaluate_csv("./pred.csv")