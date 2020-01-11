import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import SimulatedDataset
from model import Model, triplet_loss, triplet_acc

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="max number of epochs")
parser.add_argument(
    "--patience",
    type=int,
    default=50,
    help="number of epochs without improvement before stopping",
)
parser.add_argument("--bs", type=int, default=128, help="size of batches")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument(
    "--train_dir", type=str, default="data/train/bbbc038", help="training data location"
)
parser.add_argument(
    "--valid_dir",
    type=str,
    default="data/valid/bbbc038",
    help="validation data location",
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(opt):
    writer = SummaryWriter()  # tensorboard

    train_loader = DataLoader(
        SimulatedDataset(opt.train_dir), batch_size=opt.bs, shuffle=True
    )
    valid_loader = DataLoader(SimulatedDataset(opt.valid_dir), batch_size=opt.bs)

    model = Model().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    counter = 0  # epochs since improvement
    best_loss = float("inf")

    print("train_loss\tvalid_loss\tvalid_acc")

    for epoch in range(opt.n_epochs):
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

        if counter > opt.patience:
            print(f"{opt.patience} epochs without improvement, exiting")
            break


if __name__ == "__main__":
    opt = parser.parse_args()
    train(opt)
