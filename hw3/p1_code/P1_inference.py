import clip
import torch
import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm

def make_sentences(id, target):
    sentences = [
        f"This is a photo of {target}",
        f"This is not a photo of {target}",
        f"No {target}, no score.",
        f"I am sure it is a {target}",
        f"I am sure it is a real {target}",
        f"It is a real {target}", # 0.82
        f"A real {target}",
        f"It must be a real {target}"
    ]

    return sentences[id]

def draw_similarity_plot(similarity, labels, k = 5):
    values, pred_index = similarity[0].topk(k)
    
    classes, confidence = [], []
    # Print the result
    print(f"\nTop {k} predictions:\n")
    for val, id in zip(values, pred_index):
        confidence.append(100 * round(val.item(),2))
        classes.append(labels[id])
        print(f"{labels[id]:>16s}: {100 * val.item():.2f}%")
    plt.barh(classes, confidence, color='skyblue')
    plt.xlabel('Confidence')
    plt.yticks(fontsize=8)
    plt.title('It is a real {object}')
    plt.savefig(f'./fig/{classes[0]}')
    plt.close()

def inference(imgs_path, json_path, device, output_path):
    model, preprocess = clip.load("ViT-L/14", device)

    imgs_names = sorted([os.path.join(imgs_path, x) for x in os.listdir(imgs_path) if x.endswith(".png")])

    with open(json_path, 'r') as f:
        data = json.load(f)
    labels = [v for k, v in data.items()]
    
    text_tokens = torch.cat([clip.tokenize(make_sentences(5, x)) for x in labels]).to(device)
    gts = []
    preds = []
    output = []
    target_plot = [1, 200, 400]

    correct = 0
    for i, img_name in tqdm(enumerate(imgs_names)):
        img = preprocess(Image.open(img_name).convert('RGB')).unsqueeze(0).to(device)
        with torch.no_grad():
            img_feats = model.encode_image(img)
            text_feats = model.encode_text(text_tokens)

            # Pick the top 5 most similar labels for the image
            img_feats /= img_feats.norm(dim=-1, keepdim=True)
            text_feats /= text_feats.norm(dim=-1, keepdim=True)
            # print(img_feats.shape) [1, 512]
            # print(text_feats.shape) [50, 512]
            similarity = (100.0 * img_feats @ text_feats.T).softmax(dim=-1)
            values, pred_index = similarity[0].topk(1)
            
            preds.append(pred_index)
            gt_index = int(img_name.split("/")[-1].split("_")[0])
            gts.append(gt_index)
            output.append([img_name.split('/')[-1], pred_index.item()])
            
            if pred_index == gt_index:
                correct += 1
            if i in target_plot:
                print(img_name)
                draw_similarity_plot(similarity, labels)

    print(f'correctness = {(correct / len(imgs_names)):.2f}')
    df = pd.DataFrame(output, columns = ('filename', 'label'))
    df.to_csv(output_path, index = False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str)
    parser.add_argument("-j", type=str)
    parser.add_argument("-o", type=str)

    args = parser.parse_args()
    imgs_path = args.i
    json_path = args.j
    output_path = args.o

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    inference(imgs_path, json_path, device, output_path)