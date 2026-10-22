"""MNIST com a divisão do enunciado: 50 000 treino / 10 000 validação / 10 000 teste (oficial)."""
from pathlib import Path

import torch
from torch.utils.data import random_split
from torchvision import datasets, transforms

DATA_DIR = Path(__file__).resolve().parent / "data"  # baixado na primeira execução; fora do git


def load_mnist():
    """Devolve as três matrizes (N, 784) com pixels em [0, 1] e os rótulos.

    A divisão é exatamente a do enunciado. Depois, em vez de passar imagem a imagem pelo
    ToTensor (lento), divido o tensor uint8 inteiro por 255, que é o que o ToTensor faz;
    o assert confere numa imagem que o resultado é idêntico.
    """
    train_full = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transforms.ToTensor())
    test = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transforms.ToTensor())
    train, val = random_split(train_full, [50_000, 10_000],
                              generator=torch.Generator().manual_seed(42))

    def as_matrix(images):
        return images.float().div(255.0).reshape(-1, 784)

    X_train = as_matrix(train_full.data[train.indices])
    X_val = as_matrix(train_full.data[val.indices])
    X_test = as_matrix(test.data)
    assert torch.equal(X_train[0], train[0][0].reshape(-1)), "diferente do ToTensor"
    return {
        "X_train": X_train, "y_train": train_full.targets[train.indices],
        "X_val": X_val, "y_val": train_full.targets[val.indices],
        "X_test": X_test, "y_test": test.targets,
    }
