from typing import List, Union

import torch
from torch import nn

from .vgg_config import cfgs

class VGG(nn.Module):
    def __init__(self, vgg_name: str, num_classes: int = 1000) -> None:
        super().__init__()
        self.features = self._make_layers(cfgs[vgg_name])
        self.avg_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

        self._init_layers()

    def _make_layers(self, cfg: List[Union[str, int]]):
        layers = []
        in_channels = 3
        for out_channels in cfg:
            if out_channels == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif out_channels == "LRN":
                layers.append(nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2))
            else:
                if out_channels == "C1-256" or out_channels == "C1-512":
                    out_channels = int(out_channels.split("-")[1])
                    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=1))
                else:
                    out_channels = int(out_channels)
                    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
                layers.append(nn.ReLU(inplace=True))
                in_channels = out_channels
        return nn.Sequential(*layers)

    def _init_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 1e-2)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x