import torch
from config import config
from torchvision.utils import save_image
from model import *

def inference(model, device):
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        n_sample = 1000
        x_gen, x_target_step_list = model.sample(n_sample, (3, 28, 28), device, guide_w = 2)
        for i in range(n_sample // config['n_classes']): 
            for j in range(config['n_classes']):
                save_image(x_gen[i * config['n_classes'] + j], f'./figure/1000_img/{j}_{i+1:03}.png')

        n_sample = 100
        x_gen, x_target_step_list = model.sample(n_sample, (3, 28, 28), device, guide_w = 2)
        x_gen = x_gen.reshape(10, 10, 3, 28, 28)
        x_gen = x_gen.transpose(0, 1)
        x_gen = x_gen.reshape(-1, 3, 28, 28)
        save_image(x_gen, f'./figure/100_combined/total.png', nrow = 10)

        # save 6 imgs for "0" in differnet steps (500 to 1)
        step_list = [0, 5, 10, 15, 20, 25]
        for target_step in step_list:
            target_0 = x_target_step_list[target_step][0]
            save_image(torch.tensor(target_0), f'./figure/6_steps/0_{20*target_step}.png')


if __name__=="__main__":
    model = DDPM(
        backbone_model = UnetBackbone(in_channels=3, n_latent_dim=config['n_latent_dim'], n_classes=config['n_classes']),
        betas = [1e-4, 0.02],
        n_T = config['n_T'],
        device = config['device'],
        drop_prb = config['drop_prb']
    )

    model.load_state_dict(torch.load('./ckpt/model_e95.ckpt'))

    inference(model, config['device'])