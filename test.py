import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from dataset import SimulatedDataset
from model import Model, triplet_acc, triplet_loss, distance
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--bs", type=int, default=128, help="size of batches")
parser.add_argument(
    "--weights", type=str, default="models/weights.pth", help="model weights location"
)
parser.add_argument(
    "--test_dir", type=str, default="data/test/bbbc038", help="test data location"
)

parser.add_argument(
    "--display", action="store_true", help="display examples from each batch"
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def test(opt):
    loader = DataLoader(SimulatedDataset(opt.test_dir), batch_size=opt.bs)
    model = Model(nin=True).to(device)
    model.load_state_dict(torch.load(opt.weights))
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
            if opt.display:
                visualize(anchor_imgs[same_idx], same_imgs[same_idx])
                visualize(anchor_imgs[diff_idx], diff_imgs[diff_idx])
                visualize(anchor_imgs[correct_idx][:5], same_imgs[correct_idx][:5])

    print(f"loss: {total_loss/ size}")
    print(f"acc: {total_accuracy / size}")


if __name__ == "__main__":
    opt = parser.parse_args()
    test(opt)
