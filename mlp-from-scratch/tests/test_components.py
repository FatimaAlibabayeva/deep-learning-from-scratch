"""
STEP 7 (extended) --- Component tests
=======================================
The three tests in test_autograd.py cover the required B3 gradient checks
(Linear weight, hidden activation, loss). This file adds targeted checks
for everything else the pipeline depends on: broadcasting, a full MLP
gradient check, each optimizer's update rule, dropout train/eval modes,
weight decay, early stopping, the LR schedules, BatchNorm, and the
sigmoid/tanh numerical-stability fix.
"""

import numpy as np
import pytest

from src.autograd.tensor import Tensor
from src.layers import Linear, Dropout, BatchNorm1d
from src.models.perceptron import MLP, MLPBatchNorm
from src import activations
from src.losses import softmax_cross_entropy
from src.utils import gradient_check, set_seed, train_model, evaluate
from src.optimizers.sgd import SGD
from src.optimizers.momentum import Momentum
from src.optimizers.adam import Adam
from src.optimizers.rmsprop import RMSProp
from src.optimizers.schedule import StepSchedule, CosineWarmupSchedule

TOL = 1e-4


# ---------------------------------------------------------------------
# Broadcasting
# ---------------------------------------------------------------------

def test_broadcast_add_bias_gradient():
    """(N, D) + (D,) must unbroadcast the bias gradient back to shape (D,)."""
    rng = set_seed(10)
    x = Tensor(rng.normal(size=(5, 3)))
    b = Tensor(np.zeros(3))

    def f(bias):
        # rebind b's data in place so gradient_check can perturb it
        return ((x + bias) * (x + bias)).sum()

    err = gradient_check(f, b, h=1e-5)
    assert err <= TOL


def test_unbroadcast_shapes_directly():
    a = Tensor(np.ones((4, 3)))
    b = Tensor(np.zeros(3))
    out = a + b
    out.backward()
    assert b.grad.shape == (3,)
    assert np.allclose(b.grad, 4.0)


# ---------------------------------------------------------------------
# Full MLP gradient check (Linear -> ReLU -> Linear -> ReLU -> Linear)
# ---------------------------------------------------------------------

def test_gradient_check_full_mlp():
    rng = set_seed(4)
    X = rng.normal(size=(6, 8))
    y = rng.integers(0, 3, size=6)
    model = MLP(sizes=[8, 5, 4, 3], rng=rng, dropout=0.0)

    w = model.linears[1].W  # a hidden-layer weight, mid-graph

    def f(W):
        model.linears[1].W = W
        out = model(Tensor(X))
        return softmax_cross_entropy(out, y)

    err = gradient_check(f, w, h=1e-5)
    assert err <= TOL, f"relative error too high: {err}"


# ---------------------------------------------------------------------
# sigmoid / tanh: correctness + numerical stability
# ---------------------------------------------------------------------

def test_sigmoid_gradient_check():
    rng = set_seed(5)
    x = Tensor(rng.normal(size=(4, 3)))

    def f(t):
        return activations.sigmoid(t).sum()

    err = gradient_check(f, x, h=1e-5)
    assert err <= TOL


def test_tanh_gradient_check():
    rng = set_seed(6)
    x = Tensor(rng.normal(size=(4, 3)))

    def f(t):
        return activations.tanh(t).sum()

    err = gradient_check(f, x, h=1e-5)
    assert err <= TOL


def test_sigmoid_tanh_no_overflow_on_extreme_inputs():
    x = Tensor(np.array([-1e4, -1000.0, 1000.0, 1e4]))
    with np.errstate(all="raise"):
        s = activations.sigmoid(x)
        t = activations.tanh(Tensor(x.data.copy()))
    assert np.all(np.isfinite(s.data))
    assert np.all(np.isfinite(t.data))
    assert np.allclose(s.data, [0.0, 0.0, 1.0, 1.0])
    assert np.allclose(t.data, [-1.0, -1.0, 1.0, 1.0])


def test_dying_relu_zero_gradient():
    """A ReLU unit whose pre-activation is <= 0 for the whole batch must
    receive exactly zero gradient -- this IS the dying-ReLU mechanism."""
    pre_act = Tensor(np.array([[-2.0], [-0.5], [-3.0]]))
    out = pre_act.relu()
    out.backward()
    assert np.allclose(pre_act.grad, 0.0)


# ---------------------------------------------------------------------
# softmax-cross-entropy: numerical stability on extreme logits
# ---------------------------------------------------------------------

def test_softmax_cross_entropy_extreme_logits_finite():
    logits = Tensor(np.array([[1000.0, -1000.0, 0.0], [-500.0, 500.0, 0.0]]))
    y = np.array([0, 1])
    # underflow (exp of a very negative number -> 0.0) is correct softmax
    # behaviour and must NOT raise; only overflow/invalid results are bugs.
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        loss = softmax_cross_entropy(logits, y)
        loss.backward()
    assert np.isfinite(loss.data)
    assert np.all(np.isfinite(logits.grad))


# ---------------------------------------------------------------------
# Optimizers: each update rule checked against its closed-form equation
# ---------------------------------------------------------------------

def _one_param(value, grad):
    p = Tensor(np.array(value, dtype=np.float64))
    p.grad = np.array(grad, dtype=np.float64)
    return p


def test_sgd_update_rule():
    p = _one_param([1.0, -2.0], [0.1, 0.2])
    opt = SGD([p], lr=0.5, weight_decay=0.0)
    opt.step()
    assert np.allclose(p.data, [1.0 - 0.5 * 0.1, -2.0 - 0.5 * 0.2])


def test_sgd_weight_decay_shifts_update():
    p = _one_param([1.0], [0.0])  # zero raw gradient
    opt = SGD([p], lr=1.0, weight_decay=0.1)
    opt.step()
    # effective grad = 0 + 0.1*1.0 = 0.1 -> p -= 1.0*0.1
    assert np.allclose(p.data, [0.9])


def test_momentum_two_steps_match_formula():
    p = _one_param([0.0], [1.0])
    opt = Momentum([p], lr=0.1, momentum=0.9, weight_decay=0.0)
    opt.step()
    v1 = 0.9 * 0.0 + 1.0
    expected1 = 0.0 - 0.1 * v1
    assert np.allclose(p.data, [expected1])

    p.grad = np.array([1.0])
    opt.step()
    v2 = 0.9 * v1 + 1.0
    expected2 = expected1 - 0.1 * v2
    assert np.allclose(p.data, [expected2])


def test_adam_bias_correction_first_step():
    p = _one_param([0.0], [1.0])
    opt = Adam([p], lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    opt.step()
    # t=1: m_hat = m/(1-0.9^1) = 1.0 exactly (bias correction cancels the
    # (1-beta1) factor on the very first step); same for v_hat.
    m = 0.1 * 1.0
    v = 0.001 * 1.0
    m_hat = m / (1 - 0.9 ** 1)
    v_hat = v / (1 - 0.999 ** 1)
    expected = 0.0 - 0.1 * m_hat / (np.sqrt(v_hat) + 1e-8)
    assert np.allclose(p.data, [expected], atol=1e-10)


def test_rmsprop_update_rule():
    p = _one_param([0.0], [2.0])
    opt = RMSProp([p], lr=0.1, decay=0.9, eps=1e-8, weight_decay=0.0)
    opt.step()
    s = 0.1 * (2.0 ** 2)
    expected = 0.0 - 0.1 * 2.0 / (np.sqrt(s) + 1e-8)
    assert np.allclose(p.data, [expected])


# ---------------------------------------------------------------------
# LR schedules
# ---------------------------------------------------------------------

def test_step_schedule_decays_at_boundary():
    p = _one_param([0.0], [0.0])
    opt = SGD([p], lr=0.0)
    sched = StepSchedule(opt, base_lr=1.0, step_size=10, gamma=0.5)
    for _ in range(9):
        sched.step()
    assert np.isclose(opt.lr, 1.0)
    sched.step()  # 10th call
    assert np.isclose(opt.lr, 0.5)


def test_cosine_warmup_schedule_shape():
    p = _one_param([0.0], [0.0])
    opt = SGD([p], lr=0.0)
    sched = CosineWarmupSchedule(opt, base_lr=1.0, warmup_steps=10, total_steps=110)
    for _ in range(5):
        sched.step()
    assert np.isclose(opt.lr, 0.5)  # linear warm-up, halfway
    for _ in range(5):
        sched.step()
    assert np.isclose(opt.lr, 1.0, atol=1e-6)  # end of warm-up, peak lr
    for _ in range(100):
        sched.step()
    assert opt.lr < 1e-2  # cosine decay has brought it near zero


# ---------------------------------------------------------------------
# Dropout: train vs eval behaviour
# ---------------------------------------------------------------------

def test_dropout_eval_is_identity():
    rng = set_seed(7)
    d = Dropout(p=0.5, rng=rng)
    d.training = False
    x = Tensor(rng.normal(size=(100, 20)))
    out = d(x)
    assert np.allclose(out.data, x.data)


def test_dropout_train_zeros_and_scales():
    rng = set_seed(8)
    d = Dropout(p=0.5, rng=rng)
    d.training = True
    x = Tensor(np.ones((2000, 10)))
    out = d(x)
    keep_prob = 0.5
    nonzero = out.data[out.data != 0.0]
    # kept units are scaled by 1/keep_prob (inverted dropout)
    assert np.allclose(nonzero, 1.0 / keep_prob)
    frac_kept = (out.data != 0.0).mean()
    assert abs(frac_kept - keep_prob) < 0.03  # law of large numbers, 20000 draws


# ---------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------

def test_early_stopping_triggers_with_tiny_patience():
    rng = set_seed(9)
    n, d, c = 40, 6, 3
    X = rng.normal(size=(n, d))
    y = rng.integers(0, c, size=n)
    model = MLP(sizes=[d, 4, c], rng=rng, dropout=0.0)
    opt = SGD(model.parameters(), lr=0.0)  # lr=0 -> val_acc can never improve
    history = train_model(model, opt, X, y, X, y, epochs=50, batch_size=8,
                           patience=2, rng=rng)
    assert len(history["val_acc"]) <= 4  # 1 "best" epoch + patience(2) + slack


# ---------------------------------------------------------------------
# BatchNorm
# ---------------------------------------------------------------------

def test_batchnorm_train_normalizes_batch():
    rng = set_seed(11)
    bn = BatchNorm1d(num_features=5)
    x = Tensor(rng.normal(loc=3.0, scale=2.0, size=(200, 5)))
    out = bn(x)
    assert np.allclose(out.data.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(out.data.std(axis=0), 1.0, atol=1e-2)


def test_batchnorm_eval_uses_running_stats():
    rng = set_seed(12)
    bn = BatchNorm1d(num_features=4, momentum=1.0)  # momentum=1 -> running=batch stats exactly
    x = Tensor(rng.normal(loc=1.0, scale=3.0, size=(500, 4)))
    bn(x)  # one training forward pass to populate running stats
    bn.training = False
    x_eval = Tensor(rng.normal(loc=1.0, scale=3.0, size=(10, 4)))
    out = bn(x_eval)
    manual = (x_eval.data - bn.running_mean) / np.sqrt(bn.running_var + bn.eps)
    assert np.allclose(out.data, manual, atol=1e-6)


def test_batchnorm_gradient_check():
    rng = set_seed(13)
    bn = BatchNorm1d(num_features=3)
    x = Tensor(rng.normal(size=(20, 3)))

    def f(gamma):
        bn.gamma = gamma
        return (bn(x) * bn(x)).sum()

    err = gradient_check(f, bn.gamma, h=1e-5)
    assert err <= TOL


def test_mlp_batchnorm_end_to_end_runs():
    rng = set_seed(14)
    X = rng.normal(size=(16, 6))
    y = rng.integers(0, 3, size=16)
    model = MLPBatchNorm(sizes=[6, 5, 3], rng=rng)
    opt = Adam(model.parameters(), lr=1e-2)
    model.train()
    for _ in range(5):
        out = model(Tensor(X))
        loss = softmax_cross_entropy(out, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
    model.eval()
    loss_val, acc = evaluate(model, X, y)
    assert np.isfinite(loss_val)
