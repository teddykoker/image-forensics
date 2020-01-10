import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import SimulatedDataset
from model import Model, triplet_loss, triplet_acc

# TODO: these should be command line args with the following as defaults
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
bs = 128  # batch size
lr = 1e-4  # learning rate
epochs = 200  # max number of epochs
patience = 50  # number of epochs of no improvement before stopping
train_dir = "data/train/bbbc038"
valid_dir = "data/valid/bbbc038"


def train():
    writer = SummaryWriter()  # tensorboard

    train_loader = DataLoader(SimulatedDataset(train_dir), batch_size=bs, shuffle=True)
    valid_loader = DataLoader(SimulatedDataset(valid_dir), batch_size=bs)

    model = Model().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    counter = 0  # epochs since improvement
    best_loss = float("inf")

    print("train_loss\tvalid_loss\tvalid_acc")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        size = len(train_loader.dataset)

        # train step
        for i, (anchor_imgs, same_imgs, diff_imgs) in enumerate(train_loader):
            optimizer.zero_grad()
            anchor = model(anchor_imgs.to(device))
            same = model(same_imgs.to(device))
            diff = model(diff_imgs.to(device))

            loss = triplet_loss(anchor, same, diff)
            train_loss += loss.item() * anchor.size(0)
            loss.backward()  # backprop
            optimizer.step()
        train_loss /= size

        # validation step
        model.eval()
        with torch.no_grad():
            valid_loss = 0.0
            total_accuracy = 0.0
            size = len(valid_loader.dataset)
            for i, (anchor_imgs, same_imgs, diff_imgs) in enumerate(valid_loader):
                anchor = model(anchor_imgs.to(device))
                same = model(same_imgs.to(device))
                diff = model(diff_imgs.to(device))

                loss = triplet_loss(anchor, same, diff)
                valid_loss += loss.item() * anchor.size(0)
                total_accuracy += triplet_acc(anchor, same, diff) * anchor.size(0)
        valid_loss /= size
        total_accuracy /= size

        print(f"{train_loss:.3f}\t{valid_loss:.3f}\t{total_accuracy:.4f}")
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/valid", valid_loss, epoch)
        writer.add_scalar("acc/valid", total_accuracy, epoch)
        writer.flush()

        # early stopping
        if valid_loss < best_loss:
            counter = 0
            print("new best loss, saving checkpoint...")
            torch.save(model.state_dict(), f"models/checkpoint_{epoch}.pth")
            torch.save(model.state_dict(), f"models/weights.pth")
            best_loss = valid_loss
        else:
            counter += 1

        if counter > patience:
            print(f"{patience} epochs without improvement, exiting")
            break


if __name__ == "__main__":
    train()
