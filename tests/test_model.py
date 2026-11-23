import torch

from mnist_cnn.model import DigitCNN


def test_forward_shape():
    model = DigitCNN()
    out = model(torch.zeros(4, 1, 28, 28))
    assert out.shape == (4, 10)


def test_overfits_tiny_batch():
    torch.manual_seed(0)
    model = DigitCNN()
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    images, labels = torch.randn(8, 1, 28, 28), torch.randint(0, 10, (8,))
    for _ in range(30):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(images), labels)
        loss.backward()
        opt.step()
    assert loss.item() < 0.5
