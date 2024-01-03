import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import sys
import os
from PIL import Image
from tqdm import tqdm
from torchvision.utils import save_image

sys.path.insert(0, '/home/leohsu-cs/DLCV2023/hw2-LeoHsuProgrammingLab')
from UNet import UNet

'''
Reference:
1. https://zhuanlan.zhihu.com/p/565698027
2. https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py#L479
3. https://github.com/ermongroup/ddim/blob/main/runners/diffusion.py#L336
'''

def beta_scheduler(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    return betas

def get_alpha_cumprod_t(alphas_cumprod, t, shape):
    batch_size = t.shape[0]
    # gather: gather index t in dim = 0
    out = alphas_cumprod.to(t.device)[t].to(torch.float32)
    out = out.reshape(batch_size , *((1,) * (len(shape) - 1)))
    return out

def DDIM(
        model, 
        noise, 
        device,  
        batch_size=1, 
        total_timesteps=1000, 
        ddim_timesteps=50, 
        ddim_eta=0.0, 
    ):
    model = model.to(device)
    betas = beta_scheduler(n_timestep=total_timesteps, linear_start=1e-4, linear_end=2e-2)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    # print(alphas_cumprod.shape)
    
    # make ddim timestep sequence
    interval = total_timesteps // ddim_timesteps
    ddim_timestep_seq = np.asarray(list(range(0, total_timesteps, interval)))
    
    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    ddim_timestep_seq = ddim_timestep_seq + 1

    # previous sequence
    ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])
    
    # start from pure noise (for each example in the batch)
    sample_img = noise
    for i in tqdm(range(ddim_timesteps - 1, -1, -1)):
        t = torch.full((batch_size,), ddim_timestep_seq[i], device=device, dtype=torch.long)
        t_m1 = torch.full((batch_size,), ddim_timestep_prev_seq[i], device=device, dtype=torch.long)
        # print('t: ', t, 'prev_t: ', t_m1)
        # 1. get current and previous alpha_cumprod
        alpha_cumprod_t = get_alpha_cumprod_t(alphas_cumprod, t, sample_img.shape)
        alpha_cumprod_t_m1 = get_alpha_cumprod_t(alphas_cumprod, t_m1, sample_img.shape)

        # 2. predict noise using model
        pred_noise = model(sample_img, t)
        
        # 3. get the predicted x_0 in formula (4)
        pred_x0 = (sample_img - torch.sqrt((1. - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_cumprod_t)

        # 4. compute variance: "sigma_t(η)" in formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        sigmas_t = ddim_eta * torch.sqrt(
            (1 - alpha_cumprod_t_m1) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_m1))
        
        # 5. compute "direction pointing to x_t" in formula (12)
        pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_m1 - sigmas_t**2) * pred_noise 
        
        # 6. compute x_{t-1} in formula (12)
        xt_m1 = torch.sqrt(alpha_cumprod_t_m1) * pred_x0 + pred_dir_xt + sigmas_t * torch.randn_like(sample_img)

        sample_img = xt_m1
        
    pred_origin = torch.clamp(sample_img, min = -1., max = 1.)
    return pred_origin

def cal_MSE(gt_path, output_path):
    gts = sorted([os.path.join(gt_path, x) for x in os.listdir(gt_path) if x.endswith(".png")])
    outputs = sorted([os.path.join(output_path, x) for x in os.listdir(output_path) if x.endswith(".png")])

    MSEs, MSEs2 = [], []
    for gt, output in zip(gts, outputs):
        # normalize to [0, 255]
        gt = np.array(Image.open(gt))
        output = np.array(Image.open(output))
        MSE = (gt - output)**2
        MSEs.append(MSE)

        # # normalize to [0, 1] 
        # gt2 = torch.from_numpy(gt).float()
        # output2 = torch.from_numpy(output).float()
        # MSE2 = nn.MSELoss()(gt2, output2)
        # MSEs2.append(MSE2)
        # print('MSE2: ', MSE2.mean().item())

    return np.mean(MSEs)  

def interpolation(noise1, noise2, alpha, type='linear'):
    if type == 'linear':
        xt = (1 - alpha) * noise1 + alpha * noise2
    elif type == 'slerp':
        theta = torch.acos((noise1 * noise2) / (torch.norm(noise1) * torch.norm(noise2))) 
        xt = torch.sin((1 - alpha) * theta) / torch.sin(theta) * noise1 + torch.sin(alpha * theta) / torch.sin(theta) * noise2
    return xt 

def gen_10_imgs(noise_pt_list, model, device):
    model = model.to(device)
    train = True
    if train:
        for name in noise_pt_list:
            with torch.no_grad():
                noise = torch.load(name)
                tensor_out = DDIM(model, noise, device)
                output_path = f"./figure/10imgs/{name.split('/')[-1].replace('pt', 'png')}"
                save_image(tensor_out, output_path, normalize=True)

    gt_path = '/home/leohsu-cs/DLCV2023/hw2-LeoHsuProgrammingLab/hw2_data/face/GT'
    pred_path = './figure/10imgs'

    MSE = cal_MSE(gt_path, pred_path)
    print(f'MSE: {MSE}')

def gen_diff_etas(noise_pts, etas, model, device):
    model = model.to(device)
    etas = [0.0, 0.25, 0.50, 0.75, 1.0]
    tensor_out = torch.zeros((len(etas) * 4, 3, 256, 256))
    with torch.no_grad():
        for i, eta in enumerate(etas):
            for j, name in enumerate(noise_pts[:4]):
                noise = torch.load(name)
                output = DDIM(model, noise, device, ddim_eta=eta)
                tensor_out[i * 4 + j] = output
                save_image(output, f"./figure/diff_eta/eta_{eta}_noise_{name.split('/')[-1].replace('pt', 'png')}")
    print(tensor_out.shape)
    tensor_out = tensor_out.reshape(4, 5, 3, 256, 256)     
    # tensor_out = tensor_out.transpose(0, 1)
    tensor_out = tensor_out.reshape(-1, 3, 256, 256)      
    output_path = f"./figure/diff_eta/00_03.png"
    save_image(tensor_out, output_path, nrow = 4)

def gen_interpolation(noise_pt1, noise_pt2, alphas, model, device):
    model = model.to(device)
    tensor_out = torch.zeros((len(alphas), 3, 256, 256))
    type = 'slerp'
    for i in range(len(alphas)):
        with torch.no_grad():
            inter_noise = interpolation(noise_pt1, noise_pt2, alphas[i], type)
            output = DDIM(model, inter_noise, device)
            tensor_out[i] = output
            save_image(output, f"./figure/interpolation/{type}_alpha_{alphas[i]}.png")
    output_path = f"./figure/interpolation/{type}_all.png"
    save_image(tensor_out, output_path, nrow = len(alphas))

if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = '/home/leohsu-cs/DLCV2023/hw2-LeoHsuProgrammingLab/hw2_data/face/UNet.pt'
    model = UNet()
    model.load_state_dict(torch.load(model_path))

    path = '/home/leohsu-cs/DLCV2023/hw2-LeoHsuProgrammingLab/hw2_data/face/noise'
    noise_pts = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".pt")])

    
    gen_10_imgs(noise_pts, model, device)

    # etas = [0.0, 0.25, 0.50, 0.75, 1.0]
    # gen_diff_etas(noise_pts, etas, model, device)

    # noise_pt1 = torch.load(noise_pts[0])
    # noise_pt2 = torch.load(noise_pts[1])
    # alphas = [x for x in np.linspace(0, 1, 11)]
    # gen_interpolation(noise_pt1, noise_pt2, alphas, model, device)
    
