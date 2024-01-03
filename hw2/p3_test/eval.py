import pandas as pd
import argparse

def eval(gt_csv, pred_csv):
    gt = pd.read_csv(gt_csv)
    pred = pd.read_csv(pred_csv)
    all = 0
    correct = 0

    for row in gt.iterrows():
        all += 1
        name = row[1]['image_name']
        print(row[1]['label'], pred.loc[pred['image_name'] == name, 'label'].item())
        if row[1]['label'] == (pred.loc[pred['image_name'] == name, 'label'].item()):
            correct += 1

    print('result', round(correct/all, 3))

if __name__ == "__main__":
    parser =argparse.ArgumentParser()
    parser.add_argument('-gt', type=str)
    parser.add_argument('-pr', type=str)
    args = parser.parse_args()
    gt = args.gt
    pred = args.pr
    eval(gt, pred)
