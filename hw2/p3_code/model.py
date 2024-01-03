import torch.nn as nn
import torchvision.models as models

class MyResnet50(nn.Module):
    def __init__(self):
        super(MyResnet50, self).__init__()
        self.resnet50 = models.resnet50(weights = None)
        num_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Identity()
        self.label_classifier = nn.Linear(num_features, 10)

    def forward(self, x):
        l4_features = self.resnet50(x)
        x = self.label_classifier(l4_features)

        return x, l4_features

class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024), 
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512), 
            nn.BatchNorm1d(512),
            nn.ReLU(), 
            nn.Linear(512, 256), 
            nn.BatchNorm1d(256),
            nn.ReLU(), 
            nn.Linear(256, 64), 
            nn.BatchNorm1d(64), 
            nn.ReLU(), 
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        x = self.classifier(x)
        return x
