import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights,ResNet101_Weights

class Resnet50(nn.Module):
    def __init__(self, bottleneck_dim=256):
        super(Resnet50, self).__init__()
        self.feature_dim = 2048
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(model.children())[:-1])
        self.bottleneck_fc = nn.Linear(self.feature_dim, bottleneck_dim)
    def forward(self, x):
        feature = self.bottleneck_fc(torch.flatten(self.backbone(x), 1))
        return feature



class Resnet101(nn.Module):
    def __init__(self, bottleneck_dim=256):
        super(Resnet101, self).__init__()
        self.feature_dim = 2048
        model = models.resnet101(weights=ResNet101_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(model.children())[:-1])
        self.bottleneck_fc = nn.Linear(self.feature_dim, bottleneck_dim)
    def forward(self, x):
        feature = self.bottleneck_fc(torch.flatten(self.backbone(x), 1))
        return feature


class Classifier(nn.Module):
    def __init__(self, class_num, feature_dim=256):
        super(Classifier, self).__init__()
        self.classifier = torch.nn.utils.parametrizations.weight_norm(nn.Linear(feature_dim, class_num))

    def forward(self, feature):
        logics = self.classifier(feature)
        return logics