# -*- coding: utf-8 -*-

from pathlib import Path

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x


def train(model, device, loader, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        y_pred = output.argmax(dim=1, keepdim=True)
        correct += y_pred.eq(y.view_as(y_pred)).sum().item()
    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": correct / len(loader.dataset),
    }


def test(model, device, loader):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            total_loss += F.nll_loss(output, y, reduction='sum').item()
            y_pred = output.argmax(dim=1, keepdim=True)
            correct += y_pred.eq(y.view_as(y_pred)).sum().item()
    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": correct / len(loader.dataset),
    }


def main():
    data_path = Path("data")
    path = data_path / "mnist"

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    preprocess_x = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),
    ])
    additional_args = {"num_workers": 1, "pin_memory": True, } if use_cuda else {}
    train_loader = DataLoader(
        datasets.MNIST(path, train=True, download=True, transform=preprocess_x),
        batch_size=64, shuffle=True, **additional_args
    )
    test_loader = DataLoader(
        datasets.MNIST(path, train=False, transform=preprocess_x),
        batch_size=1000, shuffle=False, **additional_args
    )
    model = Net().to(device)

    optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
    with tqdm(range(1, 16)) as progress:
        for epoch in progress:
            result = train(model, device, train_loader, optimizer)
            scheduler.step()
            progress.write(str(result))
    result = test(model, device, test_loader)
    print(str(result))
    torch.save(model.state_dict(), data_path / "mnist_cnn.pt")
    # load
    # model.load_state_dict(torch.load(data_path / "mnist_cnn.pt"))


if __name__ == '__main__':
    main()
