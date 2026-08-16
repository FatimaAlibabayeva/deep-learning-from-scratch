"""
STEP 6 --- Verification tools and the training loop
======================================================
Run gradient_check BEFORE ANYTHING ELSE (required by B3) -- if it fails,
the training curve can look fine and still be wrong.
"""

import copy
import numpy as np
from src.autograd.tensor import Tensor


def set_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def gradient_check(f, x: Tensor, h: float = 1e-5) -> float:
    """Central finite-difference check of the analytic gradient in x.grad
    against numeric differentiation of f, over every element of x.data."""
    x.zero_grad()
    out = f(x)
    out.backward()
    analytic = x.grad.copy()

    numeric = np.zeros_like(x.data)
    it = np.nditer(x.data, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        orig = x.data[idx]

        x.data[idx] = orig + h
        f_plus = f(x).data.copy()

        x.data[idx] = orig - h
        f_minus = f(x).data.copy()

        x.data[idx] = orig  # restore
        numeric[idx] = (f_plus - f_minus) / (2 * h)

    denom = np.maximum(np.maximum(np.abs(analytic), np.abs(numeric)), 1.0)
    rel_err = np.abs(analytic - numeric) / denom
    return float(rel_err.max())


def evaluate(model, X, y) -> tuple[float, float]:
    from src.losses import softmax_cross_entropy

    model.eval()
    logits = model(Tensor(X))
    loss = softmax_cross_entropy(logits, y)
    preds = logits.data.argmax(axis=1)
    acc = float((preds == y).mean())
    return float(loss.data), acc


def _snapshot_params(params: list[Tensor]) -> list[np.ndarray]:
    return [p.data.copy() for p in params]


def _restore_params(params: list[Tensor], snapshot: list[np.ndarray]) -> None:
    for p, saved in zip(params, snapshot):
        p.data[:] = saved


def train_model(model, optimizer, X_train, y_train, X_val, y_val, *,
                 epochs: int, batch_size: int, patience: int,
                 rng: np.random.Generator | None = None,
                 scheduler=None) -> dict:
    from src.losses import softmax_cross_entropy
    from src.data import iterate_minibatches

    if rng is None:
        rng = np.random.default_rng(0)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    best_val_acc = -1.0
    best_params = _snapshot_params(model.parameters())
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for X_batch, y_batch in iterate_minibatches(X_train, y_train, batch_size, rng):
            logits = model(Tensor(X_batch))
            loss = softmax_cross_entropy(logits, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
            epoch_losses.append(float(loss.data))

        train_loss, train_acc = evaluate(model, X_train, y_train)
        val_loss, val_acc = evaluate(model, X_val, y_val)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = _snapshot_params(model.parameters())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    _restore_params(model.parameters(), best_params)
    return history
