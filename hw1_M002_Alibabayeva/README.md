# HW1 --- An MLP from Scratch (DLE-AI-202)

Run with:

    pip install -r requirements.txt
    python run_all.py

This runs the gradient-check tests, trains the best model
(figures/training_curves.png), compares SGD / SGD+Momentum / Adam / bonus
RMSProp+cosine-schedule (figures/optimizer_comparison.png), and prints the
final held-out test accuracy. Seed is fixed (42) inside run_all.py.

Bonus extra (not part of the required run_all.py pipeline):

    python -m experiments.bonus_batchnorm

trains a BatchNorm1d variant of the MLP and saves
figures/bonus_batchnorm.png.
