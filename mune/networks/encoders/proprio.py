import torch
import torch.nn as nn


class ProprioEncoder(nn.Module):

    def __init__(self, in_features):
        super().__init__()
        self.fc = nn.Linear(in_features, 32)

    @property
    def output_size(self):
        return 32

    def forward(self, x):
        return self.fc(x)


if __name__ == "__main__":
    encoder = ProprioEncoder(in_features=4)
    print(encoder)

    x = torch.randn((1, 4))
    print(encoder.forward(x).shape)

    print(encoder.output_size)