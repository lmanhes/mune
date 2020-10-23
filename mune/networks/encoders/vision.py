import torch
import torch.nn as nn
from collections import OrderedDict


class VisionEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv_encoder = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 32, 4, stride=2)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(32, 64, 4, stride=2)),
            ('relu2', nn.ReLU()),
            ('conv3', nn.Conv2d(64, 128, 4, stride=2)),
            ('relu3', nn.ReLU())
        ]))

    def forward(self, x):
        x = self.conv_encoder(x)
        return torch.flatten(x, start_dim=1)


if __name__ == "__main__":
    encoder = VisionEncoder()
    print(encoder)

    x = torch.randn((1, 3, 224, 224))
    print(encoder.forward(x).shape)
