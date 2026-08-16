"""BONUS 2 --- plain MLP vs MLP+BatchNorm, before/after training curves.

Not part of the required run_all.py pipeline (per the ground rules: bonus
code must not slow down or break the default run). Run standalone from the
project root:
    python -m experiments.bonus_batchnorm
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data import load_data
from src.models.perceptron import MLP, MLPBatchNorm
from src.optimizers.adam import Adam
from src.utils import train_model


EPOCHS = 40
BATCH_SIZE = 32
LR = 1e-3


def main(seed: int):
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(seed)

    rng = np.random.default_rng(seed)
    plain = MLP(sizes=[64, 32, 10], rng=rng, dropout=0.0)
    opt_plain = Adam(plain.parameters(), lr=LR)
    hist_plain = train_model(plain, opt_plain, X_train, y_train, X_val, y_val,
                              epochs=EPOCHS, batch_size=BATCH_SIZE, patience=EPOCHS, rng=rng)

    rng = np.random.default_rng(seed)
    bn = MLPBatchNorm(sizes=[64, 32, 10], rng=rng)
    opt_bn = Adam(bn.parameters(), lr=LR)
    hist_bn = train_model(bn, opt_bn, X_train, y_train, X_val, y_val,
                           epochs=EPOCHS, batch_size=BATCH_SIZE, patience=EPOCHS, rng=rng)

    epochs_plain = range(1, len(hist_plain["train_loss"]) + 1)
    epochs_bn = range(1, len(hist_bn["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(epochs_plain, hist_plain["train_loss"], label="plain MLP (train)", color="tab:blue")
    axes[0].plot(epochs_bn, hist_bn["train_loss"], label="MLP+BatchNorm (train)", color="tab:orange")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title(f"Training loss, plain vs BatchNorm (Adam, lr={LR:g})")
    axes[0].legend()

    axes[1].plot(epochs_plain, hist_plain["val_acc"], label="plain MLP (val)", color="tab:blue")
    axes[1].plot(epochs_bn, hist_bn["val_acc"], label="MLP+BatchNorm (val)", color="tab:orange")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation accuracy")
    axes[1].set_title("Validation accuracy, plain vs BatchNorm")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig("figures/bonus_batchnorm.png", dpi=150)
    plt.close(fig)
    print(f"[bonus_batchnorm] plain final val_acc={hist_plain['val_acc'][-1]:.4f}, "
          f"bn final val_acc={hist_bn['val_acc'][-1]:.4f}")
    print("[bonus_batchnorm] saved figures/bonus_batchnorm.png")


if __name__ == "__main__":
    main(42)
