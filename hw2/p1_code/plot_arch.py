from torchview import draw_graph
from model import *
from config import config
from torchvision.models import resnet18

if __name__=='__main__':
    model = UnetBackbone(in_channels=3, n_latent_dim=config['n_latent_dim'], n_classes=config['n_classes'])
    input = torch.randn(config['batch_size'], 3, 28, 28)
    label_mask = torch.bernoulli(torch.zeros((config['batch_size'], )) + 0.1)
    n_step = torch.randn(config['batch_size'])
    label = torch.randint(low=0, high=10, size=(config['batch_size'], ))

    model_graph = draw_graph(
        model, 
        input_data = [input, label, n_step, label_mask], 
        graph_name = 'try', 
        save_graph = True,
        directory = './figure/arch',
        filename = 'arch',
        roll = True
    )
    # model_graph.visual_graph