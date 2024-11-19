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
parser = Arg_parser()
args = parser.return_args()

# Parameters
p = args.p
input_dim = 2 * (p)
hidden_dim = args.hidden_dim
scale = args.init_scale
alpha = args.fraction
num_epochs = args.num_epochs
batch_size = args.batch_size
lr = args.lr
beta = args.beta
rank = args.init_rank
sparse_init = args.sparse_init
sparsity = args.sparsity


output_dim = p
X, y = generate_data(p, 'mul')
dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
train_size = int(alpha * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
criterion = nn.CrossEntropyLoss()

# Model, loss function, and optimizer
model = SimpleMLP(input_dim, hidden_dim, output_dim, scale = scale, rank = rank).to(device)
model_metainit = SimpleMLP(input_dim, hidden_dim, output_dim, scale = scale, rank = rank).to(device)
# Meta-initialization of model_metainit
metainit_with_dataset(model_metainit, criterion, train_loader, device, lr=1e-1, momentum=0.98, steps=200, eps=1e-5)
# Optimizers
if args.optimizer in ['AdamW', 'Adam']:
    optimizer = getattr(torch.optim, args.optimizer)(
        model.parameters(),
        lr=lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
    )
    optimizer_metainit = getattr(torch.optim, args.optimizer)(
        model_metainit.parameters(),
        lr=lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
    )
else:
    optimizer = getattr(torch.optim, args.optimizer)(
        model.parameters(),
        lr=lr,
        weight_decay=args.weight_decay,
    )
    optimizer_metainit = getattr(torch.optim, args.optimizer)(
        model_metainit.parameters(),
        lr=lr,
        weight_decay=args.weight_decay,
    )

# Logging metrics
metrics = {
    "model": {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []},
    "model_metainit": {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}
}

# Training loop for both models
def train_and_evaluate(model, optimizer, log_key, filter):
    grads = None
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            if args.filter == "none":
                pass
            elif args.filter == "ma":
                grads = gradfilter_ma(model, grads=grads, window_size=args.window_size, lamb=args.lamb)
            elif args.filter == "ema":
                grads = gradfilter_ema(model, grads=grads, alpha=args.alpha, lamb=args.lamb)
            else:
                raise ValueError(f"Invalid update filter type {args.filter}") 
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            running_correct += (predicted == labels).sum().item()
            total_train += labels.size(0)

        # Compute training metrics
        epoch_loss = running_loss / total_train
        epoch_acc = running_correct / total_train
        metrics[log_key]["train_loss"].append(epoch_loss)
        metrics[log_key]["train_acc"].append(epoch_acc)

        # Evaluate on test set
        model.eval()
        running_loss = 0.0
        running_correct = 0
        total_test = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                running_correct += (predicted == labels).sum().item()
                total_test += labels.size(0)

        # Compute test metrics
        epoch_loss = running_loss / total_test
        epoch_acc = running_correct / total_test
        metrics[log_key]["test_loss"].append(epoch_loss)
        metrics[log_key]["test_acc"].append(epoch_acc)

        print(f'{log_key} - Epoch {epoch + 1}/{num_epochs}, '
              f'Train Loss: {metrics[log_key]["train_loss"][-1]:.4f}, '
              f'Train Acc: {metrics[log_key]["train_acc"][-1]:.2f}, '
              f'Test Loss: {metrics[log_key]["test_loss"][-1]:.4f}, '
              f'Test Acc: {metrics[log_key]["test_acc"][-1]:.2f}')

# Train both models
train_and_evaluate(model, optimizer, "model", args.filter)
train_and_evaluate(model_metainit, optimizer_metainit, "model_metainit", args.filter)

# Plotting results
epochs_plotting = range(1, num_epochs + 1)

# Loss Plot
plt.figure()
plt.plot(epochs_plotting, metrics["model"]["train_loss"], label='Train Loss (Normal Init)', linestyle='-', color = 'blue')
plt.plot(epochs_plotting, metrics["model"]["test_loss"], label='Test Loss (Normal Init)', linestyle='--', color = 'blue')
plt.plot(epochs_plotting, metrics["model_metainit"]["train_loss"], label='Train Loss (Meta Init)', linestyle='-', color = 'orange')
plt.plot(epochs_plotting, metrics["model_metainit"]["test_loss"], label='Test Loss (Meta Init)', linestyle='--', color = 'orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xscale("log", base=10)
plt.legend()
plt.grid()
plt.title('Loss vs Epochs')
plt.savefig(f"results_metainit/compare_loss_{args.label}.png")
plt.close()

# Accuracy Plot
plt.figure()
plt.plot(epochs_plotting, metrics["model"]["train_acc"], label='Train Accuracy (Normal Init)', linestyle='-', color = 'blue')
plt.plot(epochs_plotting, metrics["model"]["test_acc"], label='Test Accuracy (Normal Init)', linestyle='--', color = 'blue')
plt.plot(epochs_plotting, metrics["model_metainit"]["train_acc"], label='Train Accuracy (Meta Init)', linestyle='-', color = 'orange')
plt.plot(epochs_plotting, metrics["model_metainit"]["test_acc"], label='Test Accuracy (Meta Init)', linestyle='--', color = 'orange')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.xscale("log", base=10)
plt.legend()
plt.grid()
plt.title('Accuracy vs Epochs')
plt.savefig(f"results_metainit/compare_accuracy_{args.label}.png")
plt.close()