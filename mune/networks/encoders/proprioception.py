import torch
import torch.nn as nn


class ProprioceptionEncoder(nn.Module):

    def __init__(self, in_features):
        super().__init__()

        if in_features <= 2:
            out_features = 8
        elif in_features > 2 and in_features < 8:
            out_features = 16
        else:
            out_features = 32

        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.fc(x)


if __name__ == "__main__":
    encoder = ProprioceptionEncoder(in_features=4)
    print(encoder)

    x = torch.randn((1, 4))
    print(encoder.forward(x).shape)