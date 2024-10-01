import math
from argparse import ArgumentParser
from itertools import permutations
import copy
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from grokfast import *

class AdamWOptim():
    def __init__(self, model, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
        self.model = model
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
        self.v = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}

    def update(self, t):
        for n, p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None:
                grad = p.grad.data.detach()

                # Momentum beta 1
                self.m[n] = self.beta1 * self.m[n] + (1 - self.beta1) * grad

                # RMS beta 2
                self.v[n] = self.beta2 * self.v[n] + (1 - self.beta2) * (grad ** 2)

                # Bias correction
                m_corr = self.m[n] / (1 - self.beta1 ** t)
                v_corr = self.v[n] / (1 - self.beta2 ** t)

                # Compute the update
                update = self.lr * (m_corr / (torch.sqrt(v_corr) + self.epsilon))
                update += self.weight_decay * self.lr * p.data.detach()

                # Update the gradient in place
                p.grad.data = update

class LookaheadOptim():
    def __init__(self, model, inner_optimizer, k=5, alpha=0.5):
        self.model = model
        self.inner_optimizer = inner_optimizer
        self.k = k
        self.alpha = alpha
        self.step_counter = 0
        self.slow_weights = {n: p.data.detach() for n, p in model.named_parameters() if p.requires_grad}
        self.fast_weights = {n: p.data.detach() for n, p in model.named_parameters() if p.requires_grad}

    def update(self, t):
        self.step_counter += 1

        # Perform inner optimizer step
        self.inner_optimizer.update(t)

        # Lookahead step
        if self.step_counter % self.k == 0:
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if p.requires_grad:
                        self.slow_weights[n] += self.alpha * (p - self.slow_weights[n])
                        p.copy_(self.slow_weights[n])
                        self.fast_weights[n] = p.data.detach()

class LambdaWarmUpScheduler:
    def __init__(self, initial_value, final_value, warmup_steps):
        self.initial_value = initial_value
        self.final_value = final_value
        self.warmup_steps = warmup_steps
        self.current_step = 0
    
    def step(self):
        if self.current_step < self.warmup_steps:
            self.current_step += 1
            warmup_ratio = self.current_step / self.warmup_steps
            return self.initial_value + warmup_ratio * (self.final_value - self.initial_value)
        else:
            return self.final_value

# scheduler for lr
class LrScheduler:
    def __init__(self, large_lr, regular_lr, warmup_steps, cutoff_steps):
        self.large_lr = large_lr
        self.regular_lr = regular_lr
        self.warmup_steps = warmup_steps
        self.cutoff_steps = cutoff_steps
        self.current_step = 0
    
    def step(self):
        if self.current_step < self.warmup_steps:
            lr = (self.large_lr / self.warmup_steps) * (self.current_step + 1)
        elif self.current_step < self.cutoff_steps:
            lr = self.large_lr
        elif self.current_step < 2 * self.cutoff_steps:
            progress = (self.current_step - self.cutoff_steps) / self.cutoff_steps
            cos_decay = 0.5 * (1 + np.cos(progress * np.pi))
            lr = self.regular_lr + (self.large_lr - self.regular_lr) * cos_decay
        else:
            lr = self.regular_lr
        self.current_step += 1
        return lr
