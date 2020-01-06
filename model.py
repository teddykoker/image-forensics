import torch
import torch.nn as nn
import torch.nn.functional as F


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, bias=True):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, bias=bias)


class ConvLayer(nn.Module):
    def __init__(self, inplanes, planes, nin=False):
        super(ConvLayer, self).__init__()
        self.bn = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        if nin:
            self.nin = conv1x1(inplanes, inplanes)
        else:
            self.nin = None
        self.conv1 = conv3x3(inplanes, planes, bias=False)
        self.conv2 = conv3x3(planes, planes)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        out = self.bn(x)

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
    def __init__(self, nin=False):
        super(Model, self).__init__()
        self.layer1 = ConvLayer(1, 16)
        self.layer2 = ConvLayer(16, 32, nin=nin)
        self.layer3 = ConvLayer(32, 64, nin=nin)
        self.layer4 = ConvLayer(64, 128, nin=nin)
        if nin:
            self.nin = conv1x1(128, 128)
        else:
            self.nin = None
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(8 * 8 * 128, 1024)
        self.fc2 = nn.Linear(1024, 128)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        if self.nin is not None:
            out = self.nin(out)
            out = self.relu(out)
        out = out.view(out.size(0), -1)  # flatten

        out = self.fc1(out)
        out = self.relu(out)  # not in original implementation
        # removed batch norm here
        out = self.fc2(out)
        return out


def _sgm(a, b):
    return F.sigmoid(1 - torch.sum(torch.abs(a - b), dim=-1))


def triplet_loss(orig, man, sim):
    sgm_same = _sgm(orig, man)
    sgm_diff = _sgm(orig, sim)
    loss = -torch.mean(torch.log(sgm_same) + torch.log(1 - sgm_diff))
    return loss


def triplet_acc(orig, man, sim):
    sgm_same = _sgm(orig, man)
    sgm_diff = _sgm(orig, sim)
    return 0.5 * (
        torch.mean((sgm_same > 0.5).float()) + torch.mean((sgm_diff <= 0.5).float())
    )
