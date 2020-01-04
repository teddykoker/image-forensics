import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, representaton_size=100):
        super(Model, self).__init__()

        # input (1, 256, 256)

        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 8, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )  # ouput (8, 127, 127)

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )  # ouput (16, 62, 62)

        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )  # ouput (16, 30, 30)

        self.fc1 = nn.Sequential(
            nn.Linear(16 * 30 * 30, 2048), nn.BatchNorm1d(2048), nn.ReLU()
        )
        self.fc2 = nn.Linear(2048, representaton_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def triplet_loss(orig, man, sim):
    sgm_same = F.sigmoid(1 - torch.sum(torch.abs(orig - man), dim=-1))
    sgm_diff = F.sigmoid(1 - torch.sum(torch.abs(orig - sim), dim=-1))
    loss = -torch.mean(torch.log(sgm_same) + torch.log(1 - sgm_diff))
    return loss


if __name__ == "__main__":
    from dataset import SimulatedDataset

    d = SimulatedDataset("data/mouse")
    og, _, _ = d[0]

    model = Model()

    x = og
    x = x.unsqueeze(0)
    print(x.shape)
    x = model(x)
    print(x.shape)

    print(triplet_loss(x, x, x))
