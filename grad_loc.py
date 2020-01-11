import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import cv2

from dataset import SimulatedDataset
from model import Model, triplet_acc, triplet_loss, distance
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument(
    "--weights", type=str, default="models/weights.pth", help="model weights location"
)
parser.add_argument(
    "--test_dir", type=str, default="data/test/bbbc038", help="test data location"
)


class GradientLocalizaton(nn.Module):
    """
    Wrapper for model that stores gradients after conv layers
    """

    def __init__(self, model):
        super(GradientLocalizaton, self).__init__()
        self.model = model
        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def activations(self, x):
        # only conv part of model
        return self.model.features(x)

    def forward(self, x):
        out = self.model.features(x)

        # important addition: register hook to store gradients
        h = out.register_hook(self.activations_hook)

        # continue linear layers
        out = out.view(out.size(0), -1)
        out = self.model.fc1(out)
        out = self.model.relu(out)
        out = self.model.fc2(out)
        return out


def impose(gradients, activations, img):
    # global pool
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # linear combination
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = np.maximum(heatmap, 0)  # ReLU
    heatmap /= torch.max(heatmap)  # normalize
    img = img.squeeze().numpy()
    img = np.stack((img,) * 3, axis=-1)  # greyscale to RGB

    # un-standardize
    # TODO: mean and std of 0.5 should be parameterized
    img = ((img * 0.5) + 0.5) * 255
    heatmap = cv2.resize(heatmap.numpy(), (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    return superimposed_img, img


def grad_loc(opt):
    loader = DataLoader(SimulatedDataset(opt.test_dir), batch_size=1, shuffle=True)
    model = Model(nin=True)
    model.load_state_dict(torch.load(opt.weights))
    loc = GradientLocalizaton(model)
    loc.eval()

    anchor, same, _ = next(iter(loader))

    dist = distance(loc(anchor), loc(same))
    print(f"distance: {dist.item()}")

    dist.backward()

    gradients = loc.gradients
    activations = loc.activations(anchor).detach()

    superimposed_anchor, anchor = impose(gradients, activations, anchor)
    cv2.imwrite("grad_anchor.png", superimposed_anchor)
    cv2.imwrite("raw_anchor.png", anchor)

    activations = loc.activations(same).detach()

    superimposed_same, same = impose(gradients, activations, same)
    cv2.imwrite("grad_same.png", superimposed_same)
    cv2.imwrite("raw_same.png", same)


if __name__ == "__main__":
    opt = parser.parse_args()
    grad_loc(opt)
