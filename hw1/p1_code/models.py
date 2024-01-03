import torchvision.models as models
import torch.nn as nn
from torchvision.models import (
    ResNeXt101_64X4D_Weights, 
    #ResNeXt101_32X8D_Weights , 
    #EfficientNet_V2_L_Weights,
    #EfficientNet_B7_Weights,
    #RegNet_Y_128GF_Weights,
    #RegNet_Y_16GF_Weights
)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 64, 64]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 512, 3, 1, 1),  # [512, 64, 64]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [512, 32, 32]

            nn.Conv2d(512, 1024, 3, 1, 1), # [1024, 32, 32]
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      #  [1024, 16, 16]

            nn.Conv2d(1024, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [512, 8, 8]

            nn.Conv2d(512, 256, 3, 1, 1), # [256, 8, 8]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [256, 4, 4]

            nn.Conv2d(256, 128, 3, 1, 1), # [128, 4, 4]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 2, 2]
        )
        self.fc = nn.Sequential(
            nn.Linear(128*2*2, 128),
            nn.ReLU(),
            nn.Linear(128, 50)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1) # 將輸出拉平，送入FC
        return self.fc(out) 
    
    def get_backbone_latent(self, x):
        out = self.cnn(x)
        return out.view(out.size()[0], -1)
    
def get_model_list(target_classes = 50):
    # model
    # ref: https://pytorch.org/vision/stable/models.html#table-of-all-available-classification-weights
    cnn_model = Classifier()

    resnext101 = models.resnext101_64x4d(weights = ResNeXt101_64X4D_Weights.IMAGENET1K_V1)
    resnext101.fc = nn.Sequential(
        nn.Linear(in_features= resnext101.fc.in_features, out_features = target_classes),
    )

    # efficientnet = models.efficientnet_v2_l(weights = EfficientNet_V2_L_Weights.IMAGENET1K_V1)
    # efficientnet = models.efficientnet_b7(weights = EfficientNet_B7_Weights.IMAGENET1K_V1)
    # efficientnet.classifier[1].out_features = target_classes

    # regnet_Y = models.regnet_y_16gf(weights = RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_E2E_V1)
    # regnet_Y.fc = nn.Sequential(
    #     nn.Linear(in_features= regnet_Y.fc.in_features, out_features = target_classes),
    # )

    model_list = [cnn_model, resnext101]

    return model_list