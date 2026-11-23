# MNIST CNN Classifier

A small PyTorch CNN for handwritten digit recognition: clean model definition, a reproducible training script with real reported metrics, and a single-image prediction CLI.

Deliberately compact - the point is a readable, correct training loop you can explain line by line, not a framework demo.

## Architecture

```
conv(1 -> 32, 3x3) + ReLU + maxpool
conv(32 -> 64, 3x3) + ReLU + maxpool
dropout(0.25) -> fc(3136 -> 128) + ReLU -> fc(128 -> 10)
```

~210k parameters. Adam, cross-entropy, MNIST normalized with its standard mean/std (0.1307, 0.3081).

## Real training run

Trained on CPU, 2 epochs over a 20,000-image subset, batch size 128:

```
epoch 1/2  train_loss=0.3616
epoch 2/2  train_loss=0.0958
test_accuracy=0.9787 (9787/10000)
saved digit-cnn.pt
```

97.87% on the full 10k test set after two epochs on a third of the training data. A full-data run (no `--subset`, 3-10 epochs) is the standard recipe and lands around 99%+.

## Usage

```bash
pip install -r requirements.txt

# full training
python -m mnist_cnn.train --epochs 3

# fast CPU demo run (what produced the numbers above)
python -m mnist_cnn.train --epochs 2 --batch-size 128 --subset 20000

# predict one image (white digit on dark background works best)
python -m mnist_cnn.predict path/to/digit.png
```

## Files

- `mnist_cnn/model.py` - the network, nothing else.
- `mnist_cnn/train.py` - training + evaluation loop; saves `digit-cnn.pt`; `--subset N` trains on the first N samples for quick runs.
- `mnist_cnn/predict.py` - loads the checkpoint and predicts one image file with a confidence score.
- `tests/test_model.py` - output shape and a tiny overfit sanity check (the model can fit 8 random images, which catches broken gradients).

## Honest limitations

- MNIST is a solved benchmark; this project shows a clean training pipeline, not research.
- `predict.py` expects MNIST-like input (single centered digit, light-on-dark); photos of digits need segmentation first.
- CPU training works fine; it uses CUDA automatically when available.

## Tests

```bash
pytest -q
```
