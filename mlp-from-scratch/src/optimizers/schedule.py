"""BONUS 1 --- Learning-rate schedules as thin wrappers over any Optimizer.

Usage:
    opt = Adam(model.parameters(), lr=1e-3)
    sched = CosineWarmupSchedule(opt, base_lr=1e-3, warmup_steps=50, total_steps=2000)
    ...
    sched.step()   # call once per optimizer.step(), updates opt.lr in place
"""

import math


class StepSchedule:
    """Multiplies lr by `gamma` every `step_size` calls."""

    def __init__(self, optimizer, base_lr: float, step_size: int, gamma: float = 0.5):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.step_size = step_size
        self.gamma = gamma
        self.t = 0
        self.optimizer.lr = base_lr

    def step(self) -> None:
        self.t += 1
        n_decays = self.t // self.step_size
        self.optimizer.lr = self.base_lr * (self.gamma ** n_decays)


class CosineWarmupSchedule:
    """Linear warm-up for `warmup_steps`, then cosine decay to ~0 by `total_steps`."""

    def __init__(self, optimizer, base_lr: float, warmup_steps: int, total_steps: int):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(total_steps, warmup_steps + 1)
        self.t = 0
        self.optimizer.lr = 0.0

    def step(self) -> None:
        self.t += 1
        if self.t <= self.warmup_steps:
            lr = self.base_lr * self.t / self.warmup_steps
        else:
            progress = (self.t - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))
        self.optimizer.lr = lr
