import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import SimulatedDataset
from model import Model, triplet_loss, triplet_acc


def train():
    writer = SummaryWriter()  # tensorboard

    # TODO: these should be command line args with the following as defaults
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bs = 128  # batch size
    lr = 1e-4  # learning rate
    epochs = 100  # number of epochs
    train_dir = "data/train/bbbc038"
    valid_dir = "data/test/bbbc038"

    train_loader = DataLoader(SimulatedDataset(train_dir), batch_size=bs, shuffle=True)
    valid_loader = DataLoader(SimulatedDataset(valid_dir), batch_size=bs)

    model = Model().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        size = len(train_loader.dataset)

        # train step
        for i, (anchor_imgs, same_imgs, diff_imgs) in enumerate(train_loader):
            optimizer.zero_grad()
            anchor = model(anchor_imgs.to(device))
            same = model(same_imgs.to(device))
            diff = model(diff_imgs.to(device))

            loss = triplet_loss(anchor, same, diff)
            total_loss += loss.item() * anchor.size(0)
            loss.backward()  # backprop
            optimizer.step()

        print(f"loss: {total_loss / size}")
        writer.add_scalar("loss/train", total_loss / size, epoch)

        # validation step
        model.eval()
        with torch.no_grad():
            total_loss = 0.0
            total_accuracy = 0.0
            size = len(valid_loader.dataset)
            for i, (anchor_imgs, same_imgs, diff_imgs) in enumerate(valid_loader):
                anchor = model(anchor_imgs.to(device))
                same = model(same_imgs.to(device))
                diff = model(diff_imgs.to(device))

                loss = triplet_loss(anchor, same, diff)
                total_loss += loss.item() * anchor.size(0)
                total_accuracy += triplet_acc(anchor, same, diff) * anchor.size(0)

        print(f"valid loss: {total_loss / size}")
        print(f"valid acc: {total_accuracy / size}")
        writer.add_scalar("loss/valid", total_loss / size, epoch)
        writer.add_scalar("acc/valid", total_accuracy / size, epoch)
        writer.flush()

        if epoch % 100 == 0:
            torch.save(model.state_dict(), f"models/weights_{epoch}.pth")

    torch.save(model.state_dict(), "models/weights.pth")


if __name__ == "__main__":
    train()
