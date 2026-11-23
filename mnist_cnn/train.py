"""Train the CNN on MNIST for a few epochs and report real test accuracy.

Usage: python -m mnist_cnn.train [--epochs 3] [--batch-size 64]
"""

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .model import DigitCNN


def run(epochs: int, batch_size: int, data_dir: str = "data", subset: int | None = None) -> float:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    if subset:
        train_set = torch.utils.data.Subset(train_set, range(subset))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(datasets.MNIST(data_dir, train=False, transform=transform), batch_size=512)

    model = DigitCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, seen = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(images)
            seen += len(images)
        print(f"epoch {epoch}/{epochs}  train_loss={total_loss / seen:.4f}")

    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            pred = model(images.to(device)).argmax(dim=1).cpu()
            correct += int((pred == labels).sum())
    accuracy = correct / len(test_loader.dataset)
    print(f"test_accuracy={accuracy:.4f} ({correct}/{len(test_loader.dataset)})")

    torch.save(model.state_dict(), "digit-cnn.pt")
    print("saved digit-cnn.pt")
    return accuracy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--subset", type=int, default=None, help="train on the first N samples only (fast demo runs)")
    args = parser.parse_args()
    run(args.epochs, args.batch_size, subset=args.subset)


if __name__ == "__main__":
    main()
