from re import L
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from grokfast import *
from optimizers import * 
from model import *
from arg_parser import Arg_parser
from tools import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters for modular arithmetic
"""
p = 97
input_dim = 2 * p
hidden_dim = 256
scale = 16.0
alpha = 0.5
batch_size = 128
lr = 4e-3
rank = [1, 2, 4, 8, 16]
output_dim = p
X, y = generate_data(p, 'mul')
dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
train_size = int(alpha * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
criterion = nn.CrossEntropyLoss()
num_runs = 5
"""

# Params for regression
input_dim = 50
num_samples = 512
hidden_dim = 2048
scale = 16.0
alpha = 0.3
batch_size = 128
lr = 1e-3
output_dim = 1
X, y, true_weights = generate_polynomial_regression_data(num_samples=num_samples, input_dim=input_dim, degree=10)
dataset = TensorDataset(X, y)
train_size = int(alpha * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
criterion = nn.MSELoss()

"""
mean_gq_baseline = compute_average_gq_multiple_runs(
    model_class=SimpleMLP,
    criterion=criterion,
    train_loader=train_loader,
    device=device,
    num_runs=num_runs,
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    scale=scale,
)
gq = [mean_gq_baseline]
# Compute the average gradient quotient over multiple runs
for r in rank:
    mean_gq = compute_average_gq_multiple_runs(
    model_class=SimpleMLP,
    criterion=criterion,
    train_loader=train_loader,
    device=device,
    num_runs=num_runs,
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    scale=scale,
    rank = r
)
    gq.append(mean_gq)

i = 0
for gq in gq:
    if i == 0:
        print(f"Mean Gradient Quotient over {num_runs} runs for baseline: {gq}")
    else:
        print(f"Mean Gradient Quotient over {num_runs} runs for rank {rank[i - 1]}: {gq}")
    i += 1
"""

"""
# below we see how metainit change the initial mean gq 
model_baseline = SimpleMLP(input_dim=input_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    scale=scale)

metainit_with_dataset(model_baseline, criterion, train_loader, device, lr=0.1, momentum=0.9, steps=500, eps=1e-5)

model_low_rank = SimpleMLP(input_dim = input_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    scale=scale, rank = 1)

metainit_with_dataset(model_low_rank, criterion, train_loader, device, lr=0.1, momentum=0.9, steps=500, eps=1e-5)

print_model_rank(model_baseline)
print_model_rank(model_low_rank)
"""

# train multiple models and plot on the same fig
# Set a fixed random seed
torch.manual_seed(42)

# Initialize two identical MLPs
model1 = SimpleMLP(input_dim = input_dim, hidden_dim = hidden_dim, output_dim = output_dim, scale = scale, rank = None)
model2 = SimpleMLP(input_dim = input_dim, hidden_dim = hidden_dim, output_dim = output_dim, scale = scale, rank = 1)
model3 = SimpleMLP(input_dim = input_dim, hidden_dim = hidden_dim, output_dim = output_dim, scale = scale, rank = None)
model4 = SimpleMLP(input_dim = input_dim, hidden_dim = hidden_dim, output_dim = output_dim, scale = scale / 10.0, rank = None)

models = [model1, model2, model3, model4]
labels = ['baseline', 'rank 1', "metainit", "small-init"]
filters = ['none', 'none', 'none', "none"]

metainit_with_dataset(model3, criterion, train_loader, device, lr=0.1, momentum=0.9, steps=2000, eps=1e-5)

# this function plots the GQ progression of the models with corresponding filters
if criterion.__class__.__name__ == "MSELoss":
    request_for_gq_regression(
        models=models,
        filters= filters,
        names= labels,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer_class=torch.optim.AdamW,  # Change optimizer if needed
        lr=lr,
        num_epochs=10000,
        device=device,
        num_gq_runs= 10, # Number of runs for mean GQ computation
        wd = 0.1
    )
else:
    request_for_gq(
        models=models,
        filters= filters,
        names= labels,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer_class=torch.optim.AdamW,  # Change optimizer if needed
        lr=lr,
        num_epochs=500,
        device=device,
        num_gq_runs= 5, # Number of runs for mean GQ computation
        wd = 0.1
    )
