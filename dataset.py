from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from pathlib import Path
from PIL import Image

from manipulations import RandomText


# See https://pytorch.org/docs/stable/torchvision/ for description of transforms
default_manipulations = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.RandomOrder(
            [
                # transforms.RandomAffine(degrees=20),
                transforms.RandomPerspective(),
                RandomText(300, 300, 10),
                transforms.RandomRotation(degrees=20),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.5),
            ]
        ),
        transforms.RandomResizedCrop(256),
    ]
)

default_transforms = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


class SimulatedDataset(Dataset):
    def __init__(
        self,
        image_dir,
        transforms=default_transforms,
        manipulations=default_manipulations,
    ):
        self.fnames = list(Path(image_dir).glob("*.png"))
        self.idx = np.arange(len(self))  # for random selection
        self.transforms = transforms
        self.manipulations = manipulations

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        original = Image.open(self.fnames[index])
        manipulated = self.manipulations(original)

        similar_idx = np.random.choice(self.idx[self.idx != index])
        similar = Image.open(self.fnames[similar_idx])

        return (
            self.transforms(original),
            self.transforms(manipulated),
            self.transforms(similar),
        )


if __name__ == "__main__":
    dataset = SimulatedDataset("data/mouse/train")
    to_img = transforms.ToPILImage()

    n = 5
    import matplotlib.pyplot as plt

    f, ax = plt.subplots(n, 3, gridspec_kw={"wspace": 0, "hspace": 0}, squeeze=True)
    for i in range(n):
        for j in range(3):
            ax[i, j].axis("off")
        og, man, sim = dataset[i]
        ax[i, 0].imshow(to_img(og), cmap="gray")
        ax[i, 1].imshow(to_img(man), cmap="gray")
        ax[i, 2].imshow(to_img(sim), cmap="gray")
    plt.tight_layout()
    plt.show()
