"""
twin training on two SimpleMLP 
with different scales
"""

# modular arithmetic in a 2-layer MLP
# = high dimensional sparse classification 

from re import L
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from grokfast import *
from optimizers import * 
from model import SimpleMLP
from model import generate_data
from model import compute_jacobian
from model import compute_ntk_batch
from model import generate_data_without_positional_labels
from arg_parser import Arg_parser

def compute_norm_effective_rank(weight_matrix):
    # Compute SVD
    U, S, V = torch.svd(weight_matrix)
    # Normalize singular values
    S_normalized = S / S.sum()
    # Compute Shannon entropy
    entropy = -(S_normalized * torch.log(S_normalized)).sum()
    # Compute effective rank
    effective_rank = torch.exp(entropy)
    rank = torch.linalg.matrix_rank(weight_matrix)
    effective_rank = effective_rank / rank
    return effective_rank.item()
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = Arg_parser()
args = parser.return_args()

# Parameters
p = args.p
input_dim = 2 * (p)
hidden_dim = args.hidden_dim
output_dim = p
scale = args.init_scale
alpha = args.fraction
num_epochs = args.num_epochs
print_interval = int(num_epochs / 100)
batch_size = args.batch_size
lr = args.lr
switch_epoch = args.switch_epoch

# Generate data and split into training and test sets
X, y = generate_data_without_positional_labels(p)
dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
train_size = int(alpha * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model, loss function, and optimizer
main_model = SimpleMLP(input_dim, hidden_dim, output_dim, scale, rank = args.init_rank).to(device)
aux_model = SimpleMLP(input_dim, hidden_dim, output_dim, scale = 0.25, rank = args.init_rank).to(device)

criterion = nn.CrossEntropyLoss()
main_optimizer = getattr(torch.optim, args.optimizer)(
        main_model.parameters(),
        lr=lr,
        weight_decay=args.weight_decay,
    )
aux_optimizer = getattr(torch.optim, args.optimizer)(
        aux_model.parameters(),
        lr=lr,
        weight_decay=args.weight_decay,
    )
scheduler = LrScheduler(large_lr=args.large_lr, regular_lr=args.lr, warmup_steps = 5, cutoff_steps=args.cutoff_steps)
main_optimizer.lr = scheduler.step()

# Training and logging
train_loss = []
test_loss = []
train_acc = []
test_acc = []
jacobian_norms = []
ntk_norms = []
emp_ntks = []
init_emp_ntk = 0
layer1_effective_ranks = []
layer2_effective_ranks = []
weight = 2.0

grads = None
for epoch in range(num_epochs):
    main_model.train()
    aux_model.train()
    running_loss = 0.0
    running_correct = 0
    total_train = 0
    jacobian_norm = 0
    ntk_norm = 0
    emp_ntk = 0
    jacobian_list_first_input = []
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
        # compute the jacobian and ntk
        jacobian_start = compute_jacobian(main_model, device, inputs)

        # optimization 
        main_optimizer.zero_grad()
        aux_optimizer.zero_grad()
        outputs_main = main_model(inputs)
        loss_main = criterion(outputs_main, labels) / float(scale**2)

        if epoch <= switch_epoch:
            print("training with aux model")
            # Forward pass through aux model
            outputs_aux = aux_model(inputs)
            loss_aux = criterion(outputs_aux, labels)

            # Backward pass through aux model
            loss_aux.backward()
            aux_optimizer.step()
            if epoch == 0:
              loss_main.backward()

            # Copy gradients from aux_model to main_model
            with torch.no_grad():
                for param_main, param_aux in zip(main_model.parameters(), aux_model.parameters()):
                    if param_main.requires_grad and param_aux.requires_grad:
                        if param_main.grad is not None and param_aux.grad is not None:
                            param_main.grad.copy_((weight * param_aux.grad + param_main.grad) / (1 + weight))
        else:
            print("training with main model")
            loss_main.backward()

        if args.filter == "none":
            pass
        elif args.filter == "ma":
            grads = gradfilter_ma(main_model, grads=grads, window_size=args.window_size, lamb=args.lamb)
        elif args.filter == "ema":
            grads = gradfilter_ema(main_model, grads=grads, alpha=args.alpha, lamb=args.lamb)
        else:
            raise ValueError(f"Invalid update filter type `{args.filter}`")
        
        main_optimizer.lr = scheduler.step()
        main_optimizer.step()
        
        running_loss += loss_main.item() * inputs.size(0)
        _, predicted = torch.max(outputs_main.data, 1)
        running_correct += (predicted == labels).sum().item()
        total_train += labels.size(0)
        # jacobian again, for the relative change
        jacobian_end = compute_jacobian(main_model, device, inputs)
        jacobian_change = torch.norm(jacobian_end - jacobian_start) 
        jacobian_norm += jacobian_change / torch.norm(jacobian_start) * inputs.size(0)

        # Extract the first row of the Jacobian for each batch
        first_row_jacobian = jacobian_start[0, :].detach().cpu()
        jacobian_list_first_input.append(first_row_jacobian)

        # ntk
        """
        ntk_end = compute_ntk_batch(device, inputs, jacobian_end_wrt_inputs)
        ntk_change = torch.norm(ntk_end - ntk_start)
        ntk_norm += ntk_change / torch.norm(ntk_start) * inputs.size(0)
        """
      
    jacobian_norms.append(jacobian_norm.item() / total_train)
    print(f'Epoch {epoch+1}, Norm of Jacobian Change: {jacobian_norm.item() / total_train}')
    epoch_jacobian = torch.vstack(jacobian_list_first_input).to(device)
    emp_ntk = compute_ntk_batch(device, epoch_jacobian)
    # print(f"the shape of the emp_ntk is {emp_ntk.shape}\n")
    if epoch == 0:
        init_emp_ntk = emp_ntk
    emp_ntk_change = torch.norm(emp_ntk - init_emp_ntk, p = 'fro').item()
    emp_ntks.append(emp_ntk_change)
    print(f'Epoch {epoch+1}, emprical NTK change: {emp_ntk_change}')
    
    # Compute and log effective ranks of two layers
    layer1_effective_rank = compute_norm_effective_rank(main_model.layer1.weight)
    layer2_effective_rank = compute_norm_effective_rank(main_model.layer2.weight)
    layer1_effective_ranks.append(layer1_effective_rank)
    layer2_effective_ranks.append(layer2_effective_rank)
    print(f'Epoch {epoch+1}, Layer1 Effective Rank: {layer1_effective_rank}')
    print(f'Epoch {epoch+1}, Layer2 Effective Rank: {layer2_effective_rank}')

    """
    ntk_norms.append(ntk_norm.item() / total_train)
    print(f'Epoch {epoch+1}, Norm of NTK Change: {ntk_norm.item() / total_train}')
    emp_ntks.append(emp_ntk.item() / total_train)
    print(f'Epoch {epoch+1}, emprical NTK change: {emp_ntk.item() / total_train}')
    """


    epoch_loss = running_loss / total_train
    epoch_acc = running_correct / total_train
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    
    main_model.eval()
    running_loss = 0.0
    running_correct = 0
    total_test = 0
    
    # testing
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
            outputs = main_model(inputs)
            loss = criterion(outputs, labels) / float(scale**2)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            running_correct += (predicted == labels).sum().item()
            total_test += labels.size(0)
    
    epoch_loss = running_loss / total_test
    epoch_acc = running_correct / total_test
    test_loss.append(epoch_loss)
    test_acc.append(epoch_acc)
    
    if epoch % print_interval == 0:
      print(f'Epoch {epoch + 1}/{num_epochs}, '
            f'Train Loss: {train_loss[-1]:.4f}, Train Acc: {train_acc[-1]:.2f}, '
            f'Test Loss: {test_loss[-1]:.4f}, Test Acc: {test_acc[-1]:.2f}')

    # Plotting
    plt.plot(train_loss, label='Train Loss')
    plt.plot(test_loss, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xscale("log", base=10)
    plt.legend()
    plt.grid()
    plt.title('Loss vs Epochs')
    plt.savefig(f"results_twin_mlp/loss_{args.label}.png")
    plt.close()


    # Plot train, test accuracy, and Jacobian
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_xscale('log', base = 10)  # Set x-axis to log scale
    ax1.set_ylabel('Accuracy (%)', color=color)
    ax1.plot(train_acc, label='Train Accuracy', color=color)
    ax1.plot(test_acc, label='Test Accuracy', linestyle='dashed', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a second y-axis for the Jacobian norms
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Jacobian Norm', color=color)
    ax2.plot(jacobian_norms, label='Jacobian relative change', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Add legends
    fig.tight_layout()  # Ensure the right y-label is not slightly clipped
    fig.legend(loc='upper left')

    plt.title('Accuracy and Jacobian Norm vs Epochs')
    plt.savefig(f"results_twin_mlp/acc_jacobian_{args.label}.png")
    plt.close()


    # Plot emprical NTK 
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_xscale('log', base = 10)  # Set x-axis to log scale
    ax1.set_ylabel('Accuracy (%)', color=color)
    ax1.plot(train_acc, label='Train Accuracy', color=color)
    ax1.plot(test_acc, label='Test Accuracy', linestyle='dashed', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a second y-axis for the Jacobian norms
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('emprical NTK', color=color)
    ax2.plot(emp_ntks, label='emprical NTK', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Add legends
    fig.tight_layout()  # Ensure the right y-label is not slightly clipped
    fig.legend(loc='upper left')

    plt.title('Accuracy and emprical NTK vs Epochs')
    plt.savefig(f"results_twin_mlp/emprical_ntk_{args.label}.png")
    plt.close()
    
    # Plot effective ranks
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_xscale('log', base = 10)  # Set x-axis to log scale
    ax1.set_ylabel('Accuracy (%)', color=color)
    ax1.plot(train_acc, label='Train Accuracy', color=color)
    ax1.plot(test_acc, label='Test Accuracy', linestyle='dashed', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a second y-axis for the Jacobian norms
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('effective ranks', color=color)
    ax2.plot(layer1_effective_ranks, label='layer 1', color=color)
    ax2.plot(layer2_effective_ranks, label='layer 2', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Add legends
    fig.tight_layout()  # Ensure the right y-label is not slightly clipped
    fig.legend(loc='upper left')

    plt.title('Accuracy and Weight Ranks vs Epochs')
    plt.savefig(f"results_twin_mlp/rank_{args.label}.png")
    plt.close()