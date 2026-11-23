"""Predict a digit from a 28x28-ish image file.

Usage: python -m mnist_cnn.predict path/to/digit.png
"""

import sys

import torch
from PIL import Image
from torchvision import transforms

from .model import DigitCNN


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    image = transform(Image.open(sys.argv[1])).unsqueeze(0)
    model = DigitCNN()
    model.load_state_dict(torch.load("digit-cnn.pt", map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        proba = torch.softmax(model(image), dim=1)[0]
    digit = int(proba.argmax())
    print(f"prediction: {digit} (confidence {float(proba[digit]):.3f})")


if __name__ == "__main__":
    main()
