import torch.nn as nn
from torchvision import models
from torchvision.models import (
    vgg16_bn,
    VGG16_BN_Weights,
    VGG16_Weights,
    ResNet50_Weights
)
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights,
)
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, FCNHead

# ref:
# 1. https://github.com/pochih/FCN-pytorch/blob/master/python/fcn.py
# 2. https://towardsdatascience.com/understand-transposed-convolutions-and-build-your-own-transposed-convolution-layer-from-scratch-4f5d97b2967


class model_A(nn.Module):
    def __init__(self, n_class = 7):
        super().__init__()
        self.relu = nn.ReLU(inplace = True)
        self.vgg16_backbone = models.vgg16_bn(weights = VGG16_BN_Weights.IMAGENET1K_V1).features
        self.fcn32 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, n_class, kernel_size=1),
        )
        self.upconv = nn.ConvTranspose2d(n_class, n_class, kernel_size = 4, stride = 10, padding = 0)
        
    def forward(self, x):
        x = self.vgg16_backbone(x)
        x = self.fcn32(x)

        return x

if __name__ == "__main__":
    model = model_A()
    print(model)

    model = deeplabv3_resnet50(weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1, weights_backbone = ResNet50_Weights.DEFAULT)
    model.aux_classifier = FCNHead(1024, 7)
    model.classifier = DeepLabHead(2048, 7)
    print(model)