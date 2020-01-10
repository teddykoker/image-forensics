import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import numpy as np
import cv2

from dataset import SimulatedDataset
from model import Model, ModelCAM, triplet_acc, triplet_loss, distance
import matplotlib.pyplot as plt


def impose(activations, img):
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = np.maximum(heatmap, 0)  # ReLU
    heatmap /= torch.max(heatmap)  # normalize
    img = img.squeeze().numpy()
    img = np.stack((img,) * 3, axis=-1)
    img = ((img * 0.5) + 0.5) * 255
    heatmap = cv2.resize(heatmap.numpy(), (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    return superimposed_img, img


def grad_cam():
    loader = DataLoader(
        SimulatedDataset("data/test/bbbc038"), batch_size=1, shuffle=True
    )
    model = Model(nin=True)
    model.load_state_dict(torch.load("models/weights.pth"))
    cam = ModelCAM(model)
    cam.eval()

    anchor, same, _ = next(iter(loader))

    dist = distance(cam(anchor), cam(same))
    print(f"dist: {dist}")

    dist.backward()

    gradients = cam.gradients
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    activations = cam.activations(anchor).detach()
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    superimposed_anchor, anchor = impose(activations, anchor)
    cv2.imwrite("cam_anchor.jpg", superimposed_anchor)
    cv2.imwrite("anchor_raw.jpg", anchor)

    activations = cam.activations(same).detach()
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    superimposed_same, same = impose(activations, same)
    cv2.imwrite("cam_same.jpg", superimposed_same)
    cv2.imwrite("same_raw.jpg", same)


if __name__ == "__main__":
    grad_cam()
