import torch

config = {
    'n_epochs': 100, 
    'n_T': 500, 
    'batch_size': 256,
    'lr': 1e-4,
    'n_latent_dim': 128,
    'drop_prb': 0.1,
    'n_classes': 10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}