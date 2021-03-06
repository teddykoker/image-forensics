import torch
import torch.nn as nn
import torch.nn.functional as F


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, bias=True):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, bias=bias)


class ConvLayer(nn.Module):
    def __init__(self, inplanes, planes, nin=False, lrn=False, lrn_size=5):
        super(ConvLayer, self).__init__()
        self.bn = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        if lrn:
            self.lrn = nn.LocalResponseNorm(lrn_size)
        else:
            self.lrn = None
        if nin:
            self.nin = conv1x1(inplanes, inplanes)
        else:
            self.nin = None
        self.conv1 = conv3x3(inplanes, planes, bias=False)
        self.conv2 = conv3x3(planes, planes)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        out = self.bn(x)

        if self.lrn is not None:
            out = self.lrn(out)

        if self.nin is not None:
            out = self.nin(out)
            out = self.relu(out)

        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool(out)
        return out


class Model(nn.Module):
    def __init__(self, nin=True, lrn=False):
        super(Model, self).__init__()
        layers = [
            ConvLayer(1, 16),
            ConvLayer(16, 32, nin=nin, lrn=lrn),
            ConvLayer(32, 64, nin=nin, lrn=lrn),
            ConvLayer(64, 128, nin=nin, lrn=lrn),
        ]
        if nin:
            layers.append(conv1x1(128, 128))
            layers.append(nn.ReLU())

        self.features = nn.Sequential(*layers)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(8 * 8 * 128, 1024)
        self.fc2 = nn.Linear(1024, 128)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)  # flatten
        out = self.fc1(out)
        out = self.relu(out)  # ReLU here not in original implementation
        # removed batch norm here
        out = self.fc2(out)
        return out


def distance(a, b):
    return torch.sum(torch.abs(a - b), dim=-1)


def _sgm(a, b):
    return F.sigmoid(1 - distance(a, b))


def triplet_loss(anchor, same, diff):
    sgm_same = _sgm(anchor, same)
    sgm_diff = _sgm(anchor, diff)
    loss = -torch.mean(torch.log(sgm_same) + torch.log(1 - sgm_diff))
    return loss


def triplet_acc(anchor, same, diff):
    sgm_same = _sgm(anchor, same)
    sgm_diff = _sgm(anchor, diff)
    return 0.5 * (
        torch.mean((sgm_same > 0.5).float()) + torch.mean((sgm_diff <= 0.5).float())
    )
