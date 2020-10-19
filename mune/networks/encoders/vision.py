import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


resize_size = {
    'efficientnet-b0': 224,
    'efficientnet-b1': 240,
    'efficientnet-b2': 260,
    'efficientnet-b3': 300,
    'efficientnet-b4': 380,
    'efficientnet-b5': 456,
    'efficientnet-b6': 528,
    'efficientnet-b7': 600,
}


class VisionEncoder(nn.Module):
    def __init__(self, model_grade=0, from_pretrained=True):
        super().__init__()
        assert model_grade in range(8)

        # EfficientNet
        self.model_name = f"efficientnet-b{model_grade}"
        self.resize_size = resize_size.get(self.model_name)
        self.input_size = (3, self.resize_size, self.resize_size)

        if from_pretrained:
            self.network = EfficientNet.from_pretrained(self.model_name)
        else:
            self.network = EfficientNet.from_name(self.model_name)

    def forward(self, x):
        assert x.size()[1:] == self.input_size, f"input size should be {self.input_size}"
        return self.network.extract_features(x).view(x.size(0), -1)


if __name__ == "__main__":
    encoder = VisionEncoder(model_grade=0, from_pretrained=False)
    print(encoder)

    x = torch.randn((1, 3, 224, 224))
    print(encoder.forward(x).shape)
