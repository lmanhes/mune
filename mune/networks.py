import torch
import torch.nn as nn
from collections import OrderedDict


class ProprioEncoder(nn.Module):
    """
    Encode proprioception sensors (in_features,) to vector (32,)
    """
    def __init__(self, in_features):
        super().__init__()
        self.fc = nn.Linear(in_features, 32)

    @property
    def output_size(self):
        return 32

    def forward(self, x):
        return self.fc(x)


class VisionEncoder(nn.Module):
    """
    Encode image observation (3, 64, 64) to vector (1024,)
    """
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


class FusionLayer(nn.Module):
    """
    Fused multiple vectors from different sensorial modalities into a unique vector (256,)
    """
    def __init__(self, in_features):
        super().__init__()
        self.fc_1 = nn.Linear(in_features, 512)
        self.fc_2 = nn.Linear(512, 256)

    def forward(self, *inputs):
        x = torch.cat(inputs, -1)
        x = self.fc_1(x)
        return self.fc_2(x)


if __name__ == "__main__":
    proprio_encoder = ProprioEncoder(in_features=4)
    vision_encoder = VisionEncoder()
    fusion_layer = FusionLayer(in_features=proprio_encoder.output_size+vision_encoder.output_size)

    x_proprio = torch.randn((1, 4))
    x_vision = torch.randn((1, 3, 64, 64))

    h_proprio = proprio_encoder(x_proprio)
    h_vision = vision_encoder(x_vision)

    fused_state = fusion_layer(h_proprio, h_vision)

    print(fused_state.size())