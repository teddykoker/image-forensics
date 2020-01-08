import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from dataset import SimulatedDataset
from model import Model, triplet_acc, triplet_loss, distance
import matplotlib.pyplot as plt


def visualize(lefts, rights):
    f, ax = plt.subplots(
        lefts.shape[0], 2, gridspec_kw={"wspace": 0, "hspace": 0}, squeeze=True
    )
    for i in range(lefts.shape[0]):
        ax[i, 0].axis("off")
        ax[i, 1].axis("off")
        ax[i, 0].imshow(lefts[i, 0].cpu().numpy(), cmap="gray")
        ax[i, 1].imshow(rights[i, 0].cpu().numpy(), cmap="gray")
    plt.tight_layout()
    plt.show()


def test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(SimulatedDataset("data/test/bbbc038"), batch_size=128)
    model = Model(nin=True).to(device)
    model.load_state_dict(torch.load("models/weights.pth"))
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        total_accuracy = 0.0
        size = len(loader.dataset)
        for i, (anchor_imgs, same_imgs, diff_imgs) in enumerate(loader):
            anchor = model(anchor_imgs.to(device))
            same = model(same_imgs.to(device))
            diff = model(diff_imgs.to(device))

            loss = triplet_loss(anchor, same, diff)
            total_loss += loss.item() * anchor.size(0)
            total_accuracy += triplet_acc(anchor, same, diff) * anchor.size(0)

            same_idx = distance(anchor, same) > 1.0
            diff_idx = distance(anchor, diff) <= 1.0
            correct_idx = distance(anchor, same) < 1.0
            visualize(anchor_imgs[same_idx], same_imgs[same_idx])
            visualize(anchor_imgs[diff_idx], diff_imgs[diff_idx])
            visualize(anchor_imgs[correct_idx][:5], same_imgs[correct_idx][:5])

    print(f"loss: {total_loss/ size}")
    print(f"acc: {total_accuracy / size}")


if __name__ == "__main__":
    test()
