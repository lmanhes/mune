import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionLayer(nn.Module):

    def __init__(self, in_features):
        super().__init__()

        self.fc_1 = nn.Linear(in_features, 512)
        self.fc_2 = nn.Linear(512, 128)

    def forward(self, *inputs):
        x = torch.cat(inputs, -1)
        x = F.relu(self.fc_1(x))
        return self.fc_2(x)


if __name__ == "__main__":
    fusion_layer = FusionLayer(in_features=86528)
    print(fusion_layer)

    x = torch.randn((1, 28))
    y = torch.randn((1, 86500))
    print(fusion_layer.forward(x, y).shape)