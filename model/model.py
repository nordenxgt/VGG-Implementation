from typing import List, Union

import torch
from torch import nn

from vgg_config import cfgs

class VGG(nn.Module):
    def __init__(self, vgg_name: str, num_classes: int = 1000) -> None:
        super().__init__()
        self.features = self._make_layers(cfgs[vgg_name])
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
        for x in cfg:
            if x == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif x == "LRN":
                layers.append(nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2))
            else:
                if x == "C1-256" or x == "C1-512":
                    x = int(x.split("-")[1])
                    layers.append(nn.Conv2d(in_channels, x, kernel_size=1, stride=1, padding=1))
                else:
                    x = int(x)
                    layers.append(nn.Conv2d(in_channels, x, kernel_size=3, stride=1, padding=1))
                layers.append(nn.ReLU(inplace=True))
                in_channels = x
        return nn.Sequential(*layers)

    def _init_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 1e-2)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x