import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import SimulatedDataset
from model import Model, triplet_loss, triplet_acc


def train(epochs=1000):
    writer = SummaryWriter()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(
        SimulatedDataset("data/train/bbbc038"), batch_size=128, shuffle=True
    )
    test_loader = DataLoader(SimulatedDataset("data/test/bbbc038"), batch_size=128)

    model = Model(nin=True).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        size = len(train_loader.dataset)
        for i, (origs, mans, sims) in enumerate(train_loader):
            origs = origs.to(device)
            mans = mans.to(device)
            sims = sims.to(device)
            optimizer.zero_grad()

            origs_repr = model(origs)
            mans_repr = model(mans)
            sims_repr = model(sims)

            loss = triplet_loss(origs_repr, mans_repr, sims_repr)
            total_loss += loss.item() * origs.size(0)
            loss.backward()
            optimizer.step()

        print(f"loss: {total_loss / size}")
        writer.add_scalar("loss/train", total_loss / size, epoch)

        model.eval()
        with torch.no_grad():
            total_loss = 0.0
            total_accuracy = 0.0
            size = len(test_loader.dataset)
            for i, (origs, mans, sims) in enumerate(test_loader):
                origs = origs.to(device)
                mans = mans.to(device)
                sims = sims.to(device)

                origs_repr = model(origs)
                mans_repr = model(mans)
                sims_repr = model(sims)

                loss = triplet_loss(origs_repr, mans_repr, sims_repr)
                total_loss += loss.item() * origs.size(0)
                total_accuracy += triplet_acc(
                    origs_repr, mans_repr, sims_repr
                ) * origs.size(0)

        print(f"test loss: {total_loss / size}")
        print(f"test acc: {total_accuracy / size}")
        writer.add_scalar("loss/test", total_loss / size, epoch)
        writer.add_scalar("acc/test", total_accuracy / size, epoch)


if __name__ == "__main__":
    train()
