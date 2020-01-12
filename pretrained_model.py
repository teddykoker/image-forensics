import torch
import torch.nn as nn
from torch.utils import model_zoo
from torchvision.models.resnet import model_urls, ResNet, Bottleneck


def _pretrained_resnet50(inchans=3):
    resnet = ResNet(Bottleneck, [3, 4, 6, 3])
    state_dict = model_zoo.load_url(model_urls["resnet50"])
    if inchans == 1:
        # replace first conv layer with one channel
        # combine weights of 3 input channels so greyscale (1 channel) images
        # can be used

        # original code: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L141
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        conv1_weight = state_dict["conv1.weight"]
        state_dict["conv1.weight"] = conv1_weight.sum(dim=1, keepdim=True)
    resnet.load_state_dict(state_dict)
    return resnet


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class PretrainedModel(nn.Module):
    def __init__(self):
        super(PretrainedModel, self).__init__()
        resnet = _pretrained_resnet50(inchans=1)
        in_features = resnet.fc.in_features

        # resnet without fully connected layers
        self.features = nn.Sequential(*list(resnet.children())[:-2])

        self.head = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            Flatten(),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
        )

    def freeze(self):
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.features.parameters():
            param.requires_grad = True

    def forward(self, x):
        out = self.features(x)
        out = self.head(out)
        return out
