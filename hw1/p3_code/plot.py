import numpy as np
import imageio
import os

def pred2image(pred, name, out_path):
    pred = pred.numpy()
    pred_img = np.zeros((512, 512, 3), dtype=np.uint8)
    pred_img[pred == 0] = [0, 255, 255]
    pred_img[pred == 1] = [255, 255, 0]
    pred_img[pred == 2] = [255, 0, 255]
    pred_img[pred == 3] = [0, 255, 0]
    pred_img[pred == 4] = [0, 0, 255]
    pred_img[pred == 5] = [255, 255, 255]
    pred_img[pred == 6] = [0, 0, 0]
    imageio.imwrite(os.path.join(out_path, name.replace('sat.jpg', f'mask.png')), pred_img)

