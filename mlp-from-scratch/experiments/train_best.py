"""STEP 8(a) --- Train the best model, save figures/training_curves.png"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data import load_data
from src.models.perceptron import MLP
from src.optimizers.adam import Adam
from src.utils import set_seed, evaluate, train_model
from src.autograd.tensor import Tensor


def main(seed: int) -> float:
    rng = set_seed(seed)
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(seed)

    model = MLP(sizes=[64, 32, 10], rng=rng, dropout=0.1)
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    history = train_model(
        model, optimizer, X_train, y_train, X_val, y_val,
        epochs=200, batch_size=32, patience=15, rng=rng,
    )

    test_loss, test_acc = evaluate(model, X_test, y_test)

    epochs_range = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(epochs_range, history["train_loss"], label="train loss")
    axes[0].plot(epochs_range, history["val_loss"], label="val loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs_range, history["train_acc"], label="train acc")
    axes[1].plot(epochs_range, history["val_acc"], label="val acc")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    fig.suptitle(f"final test accuracy = {test_acc:.4f}")
    fig.tight_layout()
    fig.savefig("figures/training_curves.png", dpi=150)
    plt.close(fig)

    print(f"[train_best] final test accuracy: {test_acc:.4f} (loss={test_loss:.4f})")
    return test_acc


if __name__ == "__main__":
    import os
    main(42)
