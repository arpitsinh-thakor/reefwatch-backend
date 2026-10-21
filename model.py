import torch.nn as nn
import timm


class AttnHead(nn.Module):
    """Adaptive average/max pooling head with squeeze-excite-like channel attention."""
    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_features, in_features // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features // 8, in_features, 1),
            nn.Sigmoid(),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.bn = nn.BatchNorm1d(in_features * 2)
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(in_features * 2, 512)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(512)
        self.drop2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x * self.se(x)
        avg = self.avgpool(x).flatten(1)
        mx = self.maxpool(x).flatten(1)
        x = self.bn(__import__("torch").cat([avg, mx], dim=1))
        x = self.drop(x)
        x = self.relu(self.fc1(x))
        x = self.bn2(x)
        x = self.drop2(x)
        return self.fc2(x)


class CoralNet(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int, dropout: float = 0.3):
        super().__init__()
        # pretrained=False avoids downloading ImageNet weights; the checkpoint supplies trained weights.
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, num_classes=0, global_pool=""
        )
        self.head = AttnHead(self.backbone.num_features, num_classes, dropout=dropout)

    def forward(self, x):
        return self.head(self.backbone.forward_features(x))


def build_model(name: str, num_classes: int = 3, dropout: float = 0.3):
    return CoralNet(name, num_classes, dropout=dropout)
