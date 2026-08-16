"""STEP 8(b) --- SGD vs Momentum vs Adam (+ BONUS: RMSProp with a cosine
LR schedule), same seed/model/budget, val-accuracy curves on one axes."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data import load_data
from src.models.perceptron import MLP
from src.optimizers.sgd import SGD
from src.optimizers.momentum import Momentum
from src.optimizers.adam import Adam
from src.optimizers.rmsprop import RMSProp
from src.optimizers.schedule import CosineWarmupSchedule
from src.utils import set_seed, train_model


EPOCHS = 60
BATCH_SIZE = 32


def _fresh_model(seed):
    rng = np.random.default_rng(seed)  # same architecture-init seed every run
    return MLP(sizes=[64, 32, 10], rng=rng, dropout=0.0), rng


def main(seed: int):
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(seed)

    curves = {}

    # --- SGD ---
    model, rng = _fresh_model(seed)
    opt = SGD(model.parameters(), lr=0.1)
    hist = train_model(model, opt, X_train, y_train, X_val, y_val,
                        epochs=EPOCHS, batch_size=BATCH_SIZE, patience=EPOCHS, rng=rng)
    curves["SGD (lr=0.1)"] = hist["val_acc"]

    # --- SGD + Momentum ---
    model, rng = _fresh_model(seed)
    opt = Momentum(model.parameters(), lr=0.05, momentum=0.9)
    hist = train_model(model, opt, X_train, y_train, X_val, y_val,
                        epochs=EPOCHS, batch_size=BATCH_SIZE, patience=EPOCHS, rng=rng)
    curves["SGD+Momentum (lr=0.05)"] = hist["val_acc"]

    # --- Adam ---
    model, rng = _fresh_model(seed)
    opt = Adam(model.parameters(), lr=1e-3)
    hist = train_model(model, opt, X_train, y_train, X_val, y_val,
                        epochs=EPOCHS, batch_size=BATCH_SIZE, patience=EPOCHS, rng=rng)
    curves["Adam (lr=1e-3)"] = hist["val_acc"]

    # --- BONUS: RMSProp + cosine-warmup LR schedule ---
    model, rng = _fresh_model(seed)
    opt = RMSProp(model.parameters(), lr=1e-3)
    n_batches_per_epoch = int(np.ceil(len(X_train) / BATCH_SIZE))
    sched = CosineWarmupSchedule(opt, base_lr=1e-3, warmup_steps=5 * n_batches_per_epoch,
                                  total_steps=EPOCHS * n_batches_per_epoch)
    hist = train_model(model, opt, X_train, y_train, X_val, y_val,
                        epochs=EPOCHS, batch_size=BATCH_SIZE, patience=EPOCHS, rng=rng,
                        scheduler=sched)
    curves["RMSProp+cosine, bonus (base lr=1e-3)"] = hist["val_acc"]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, vals in curves.items():
        ax.plot(range(1, len(vals) + 1), vals, label=name)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation accuracy")
    ax.set_title("Optimizer comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/optimizer_comparison.png", dpi=150)
    plt.close(fig)

    for name, vals in curves.items():
        print(f"[compare_optimizers] {name}: final val_acc={vals[-1]:.4f}, best={max(vals):.4f}")


if __name__ == "__main__":
    import os
    main(42)
