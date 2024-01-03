from torchvision import models
import torch.nn as nn
import torch

class MyResnet50(nn.Module):
    def __init__(self, backbone_ckpt = None, setting = "C"):
        super().__init__()
        self.resnet50 = models.resnet50(weights = None)
        if backbone_ckpt != None:
            self.resnet50.load_state_dict(torch.load(backbone_ckpt))
        if setting == "D" or setting == "E":
            for param in self.resnet50.parameters():
                param.requires_grad = False
        
        self.fc = nn.Sequential(
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 65)
        )
        
    def forward(self, x):
        x = self.resnet50(x)
        x = self.fc(x)
        return x