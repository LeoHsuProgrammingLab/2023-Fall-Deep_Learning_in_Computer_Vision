import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm


# Reference: 
# 1. https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py
# 2. https://github.com/cloneofsimo/minDiffusion
# 3. https://medium.com/ai-blog-tw/%E9%82%8A%E5%AF%A6%E4%BD%9C%E9%82%8A%E5%AD%B8%E7%BF%92diffusion-model-%E5%BE%9Eddpm%E7%9A%84%E7%B0%A1%E5%8C%96%E6%A6%82%E5%BF%B5%E7%90%86%E8%A7%A3-4c565a1c09c
# 4. https://github.com/TeaPearce/Conditional_Diffusion_MNIST/blob/main/script.py
# 5. https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L306
# 6. https://tree.rocks/make-diffusion-model-from-scratch-easy-way-to-implement-quick-diffusion-model-e60d18fd0f2e
# 7. https://huggingface.co/blog/annotated-diffusion


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_residual: False, groups = 8):
        super(ConvBlock, self).__init__()
        self.in_out_same_dim = (in_channels == out_channels)
        self.is_residual = is_residual
        self.conv1_1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0)  
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )
        self.group_norm = nn.GroupNorm(groups, out_channels)
    
    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        if(self.is_residual):
            if self.in_out_same_dim:
                return x + x2
            else: 
                return self.conv1_1(x) + x2
        else:
            return x2

class UnetDownsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDownsample, self).__init__()
        self.block1 = nn.Sequential(
            ConvBlock(in_channels, out_channels, is_residual = True),
            nn.MaxPool2d(2, 2, 0)
        )
    
    def forward(self, x):
        return self.block1(x)
    
class UnetUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUpsample, self).__init__()
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2, 0),
            ConvBlock(out_channels, out_channels, is_residual = True), 
            ConvBlock(out_channels, out_channels, is_residual = True)
        )
    
    def forward(self, x, shortcut):
        x = torch.cat([x, shortcut], dim = 1)
        return self.block1(x)

class LatentLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LatentLayer, self).__init__()
        self.in_channels = in_channels
        self.block1 = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, x):
        x = x.reshape(-1, self.in_channels)
        return self.block1(x)

class UnetBackbone(nn.Module): # Combine different modules to backbone
    def __init__(self, in_channels, n_latent_dim, n_classes=10):
        super(UnetBackbone, self).__init__()
        self.in_channels = in_channels
        self.n_latent_dim = n_latent_dim
        self.n_classes = n_classes

        self.init_block = ConvBlock(in_channels, n_latent_dim, is_residual = True)
        self.downsample1 = UnetDownsample(n_latent_dim, n_latent_dim)
        self.downsample2 = UnetDownsample(n_latent_dim, n_latent_dim * 2)

        self.extract_latent = nn.Sequential(
            nn.AvgPool2d(7),
            nn.SiLU()
        )
        # For training step features
        self.step_latent1 = LatentLayer(1, n_latent_dim)
        self.step_latent2 = LatentLayer(1, n_latent_dim * 2)
        # For label features
        self.label_latent1 = LatentLayer(n_classes, n_latent_dim)
        self.label_latent2 = LatentLayer(n_classes, n_latent_dim * 2)

        self.upsample_init1 = nn.Sequential(
            nn.ConvTranspose2d(n_latent_dim * 2, n_latent_dim * 2, 7, 7, 0),
            nn.BatchNorm2d(n_latent_dim * 2),
            nn.SiLU()
        )

        self.upsample_init2 = nn.Sequential(
            nn.ConvTranspose2d(n_latent_dim * 6, n_latent_dim * 2, 7, 7, 0),
            nn.BatchNorm2d(n_latent_dim * 2),
            nn.SiLU()
        )

        self.upsample1 = UnetUpsample(n_latent_dim * 2, n_latent_dim)
        self.upsample2 = UnetUpsample(n_latent_dim * 4, n_latent_dim)
        
        self.out = nn.Sequential(
            nn.Conv2d(n_latent_dim * 2, n_latent_dim, 3, 1, 1),
            nn.GroupNorm(8, n_latent_dim),
            nn.SiLU(),
            nn.Conv2d(n_latent_dim, self.in_channels, 3, 1, 1)
        )
    
    def forward(self, x, label, n_step, label_mask):
        # print('x:', x.shape)
        # print('label:', label.shape)
        # print('n_step:', n_step.shape)
        # print('label_mask:', label_mask.shape)
        x = self.init_block(x)
        down_latent1 = self.downsample1(x) # n_latent_dim
        down_latent2 = self.downsample2(down_latent1) # n_latent_dim * 2
        hidden_layer = self.extract_latent(down_latent2) # n_latent_dim * 2

        # label to one hot
        label = nn.functional.one_hot(label, self.n_classes).float() 

        # label_mask is for drop_out if label_mask[:, i] == 1
        label_mask = label_mask[:, None] # turn to (batch_size, 1)
        label_mask = label_mask.repeat(1, self.n_classes) # turn to (batch_size, n_classes) for n_classes one hot
        label_mask = (-1 * (1 - label_mask)) 
        label = label_mask * label

        # embedding label & step
        label_latent1 = self.label_latent1(label).reshape(-1, self.n_latent_dim, 1, 1)
        label_latent2 = self.label_latent2(label).reshape(-1, self.n_latent_dim * 2, 1, 1)
        step_latent1 = self.step_latent1(n_step).reshape(-1, self.n_latent_dim, 1, 1)
        step_latent2 = self.step_latent2(n_step).reshape(-1, self.n_latent_dim * 2, 1, 1)

        # concat
        up_latent_init = torch.cat([hidden_layer, label_latent2, step_latent2], dim = 1)
        up_latent_init = self.upsample_init2(up_latent_init)
        up_latent1 = self.upsample2(up_latent_init * label_latent2 + step_latent2, down_latent2)
        up_latent2 = self.upsample1(up_latent1 * label_latent1 + step_latent1, down_latent1)
        out = self.out(torch.cat([up_latent2, x], 1))

        return out
    
def ddpm_scheduler(beta1, beta2, t):
    # Predefine the DDPM sampling schedule in training process
    # (beta1: the first beta value) < (beta2: the second beta value) < 1
    # t: current step

    # linear scheduling
    beta_t = (beta2 - beta1) * torch.arange(0, t+1, dtype=torch.float32) / t + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alpha_bar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
    oneover_sqrt_alpha_t = 1 / torch.sqrt(alpha_t)

    sqrt_1m_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
    beta_over_sqrt_1m_alpha_bar_t = beta_t / sqrt_1m_alpha_bar_t

    return {
        'alpha_t': alpha_t, # 1 - beta_t
        'oneover_sqrt_alpha_t': oneover_sqrt_alpha_t,
        'sqrt_beta_t': sqrt_beta_t,
        'alpha_bar_t': alpha_bar_t, # exp( cumulative sum of log_alpha_t )
        'sqrt_alpha_bar_t': sqrt_alpha_bar_t, # abar_t ** 0.5
        'sqrt_1m_alpha_bar_t': sqrt_1m_alpha_bar_t,
        'beta_over_sqrt_1m_alpha_bar_t': beta_over_sqrt_1m_alpha_bar_t
    }

class DDPM(nn.Module):
    def __init__(self, backbone_model, betas, n_T, device, drop_prb = 0.1):
        super(DDPM, self).__init__()
        self.model = backbone_model.to(device)
        self.device = device
        self.drop_prb = drop_prb
        self.n_T = n_T # total steps
        self.criterion = nn.MSELoss()

        for k, v in ddpm_scheduler(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)
        
    def forward(self, x, label):
        step_id_mask = torch.randint(1, self.n_T, (x.shape[0], )).to(self.device) # random sample from 1 to n_T steps
        noise = torch.randn(*x.shape).to(self.device) # noise for the shape of tensor x, cracking for different inputs
        x_t = (self.sqrt_alpha_bar_t[step_id_mask, None, None, None] * x + 
               self.sqrt_1m_alpha_bar_t[step_id_mask, None, None, None] * noise)
        label_mask = torch.bernoulli(torch.zeros(*label.shape) + self.drop_prb).to(self.device)
        random_step = step_id_mask / self.n_T

        return self.criterion(noise, self.model(x_t, label, random_step, label_mask))
    
    def sample(self, n_sample, size, device, guide_w = 0.0):
        x_i = torch.randn(n_sample, *size).to(device)
        label_i = torch.arange(0, 10).to(device)
        label_i = label_i.repeat(n_sample // label_i.shape[0]) # randomly choose n_sample labels

        # Don't mask during inference
        label_mask = torch.zeros(*label_i.shape).to(device)

        # Double the mask size for guided Weighted Sampling between t step and t-1 step
        label_i = label_i.repeat(2)
        label_mask = label_mask.repeat(2)
        label_mask[n_sample:] = 1.0

        x_i_step_list = [] # store the generated images

        for i in range(self.n_T, 0, -1):
            print(f'step {i} starts', end='\r')
            # step_i = i / target_n_sample
            step_i = torch.tensor([i / self.n_T]).to(device)
            step_i = step_i.repeat(n_sample, 1, 1, 1)

            x_i = x_i.repeat(2, 1, 1, 1)
            step_i = step_i.repeat(2, 1, 1, 1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            eps = self.model(x_i, label_i, step_i, label_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            # guided Weighted Sampling: Separate the conditional and unconditional data
            eps = (1 + guide_w) * eps1 - guide_w * eps2
            x_i = x_i[:n_sample]
            x_i = (
                self.oneover_sqrt_alpha_t[i] * (x_i - eps * self.beta_over_sqrt_1m_alpha_bar_t[i]) +
                self.sqrt_beta_t[i] * z
            )
            
            if i == self.n_T or i % 20 == 0 or i == 1:
                x_i_step_list.append(x_i.detach().cpu().numpy())

        print('len of list: ', len(x_i_step_list))
        x_i_step_list = np.array(x_i_step_list)
        return x_i, x_i_step_list