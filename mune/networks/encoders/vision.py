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
            ('relu3', nn.ReLU()),
            ('conv4', nn.Conv2d(128, 256, 4, stride=2)),
            ('relu4', nn.ReLU())
        ]))

    @property
    def output_size(self):
        return self(torch.randn((1, 3, 64, 64))).size(-1)

    def forward(self, x):
        x = self.conv_encoder(x)
        return torch.flatten(x, start_dim=1)


if __name__ == "__main__":
    encoder = VisionEncoder()
    print(encoder)

    x = torch.randn((1, 3, 64, 64))
    print(encoder.forward(x).shape)

    print(encoder.output_size)
