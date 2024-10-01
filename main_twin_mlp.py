"""
twin training on two SimpleMLP 
with different scales
"""

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

def compute_cosine_similarity(matrix1, matrix2):
    # Flatten the matrices into vectors
    vec1 = matrix1.view(-1)
    vec2 = matrix2.view(-1)
    
    # Compute the dot product of the two vectors
    dot_product = torch.dot(vec1, vec2)
    
    # Compute the magnitude (Euclidean norm) of each vector
    magnitude_vec1 = torch.norm(vec1)
    magnitude_vec2 = torch.norm(vec2)
    
    # Compute the cosine similarity
    cosine_sim = dot_product / (magnitude_vec1 * magnitude_vec2)
    
    return cosine_sim.item()  # Convert to a Python float for readability

def low_rank_approximation(matrix, rank):
    # Perform SVD on the attention weights matrix
    U, S, V = torch.svd(matrix)
    # Retain only the top 'rank' singular values
    S = torch.diag(S[:rank])
    U = U[:, :rank]
    V = V[:, :rank]
    # Recompose the matrix with reduced rank
    low_rank_matrix = torch.mm(U, torch.mm(S, V.t()))
    return low_rank_matrix

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
batch_size = args.batch_size
lr = args.lr
beta = args.beta
rank = args.init_rank
switch_epoch = args.switch_epoch
aligned = args.alignment
delta_rank = args.switch_to_rank
direction_searching_method = args.direction_searching_method

# some hard-coded hyper-params 
scale_gap = 20.0
amp_factor = 1.0
top_k_percent = 5

# Generate data and split into training and test sets
X, y = generate_data_without_positional_labels(p)
dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
train_size = int(alpha * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model, loss function, and optimizer
aux_model = SimpleMLP(input_dim, hidden_dim, output_dim, scale = scale / scale_gap, rank = rank).to(device)
if aligned:
    model = SimpleMLP(input_dim, hidden_dim, output_dim, scale = scale, rank = rank).to(device)
    model.load_state_dict(aux_model.state_dict())
    model.layer1.weight.data *= scale_gap
    model.layer2.weight.data *= scale_gap
else:
    model = SimpleMLP(input_dim, hidden_dim, output_dim, scale, rank = rank).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = getattr(torch.optim, args.optimizer)(
        model.parameters(),
        lr=lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
    )
aux_optimizer = getattr(torch.optim, args.optimizer)(
        aux_model.parameters(),
        lr=lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
    )
# scheduler = LrScheduler(large_lr=args.lr, regular_lr=args.lr, warmup_steps = 10, cutoff_steps=args.cutoff_steps)

num_singular_values = 10

# Initialize a 2D array to store the singular values
singular_values = np.zeros((num_singular_values, num_epochs))


# Training and logging
train_loss = []
test_loss = []
train_acc = []
test_acc = []

"""
jacobian_norms = []
emp_ntks = []
emp_ntk_similarity_log = []
emp_ntk_copy = []
init_emp_ntk = 0
emp_ntk = 0
"""

layer1_effective_ranks = []
layer2_effective_ranks = []
layer1_spec_norm = []
layer2_spec_norm = []
layer1_spec_norm_init = torch.linalg.norm(model.layer1.weight, ord = 2).item()
layer2_spec_norm_init = torch.linalg.norm(model.layer2.weight, ord = 2).item()

"""
update_ranks_layer1 = []
update_ranks_layer2 = []
update_ranks_from_init_layer1 = []
update_ranks_from_init_layer2 = []
"""

"""
--------- train aux_model for switch_epoch steps to find an initialization for the main model ------------
"""
for epoch in range(switch_epoch):
    aux_model.train()
    running_correct = 0
    total_train = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        aux_optimizer.zero_grad()
        outputs = aux_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        aux_optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        running_correct += (predicted == labels).sum().item()
        total_train += labels.size(0)

    epoch_train_acc = running_correct / total_train
    aux_model.eval()

    running_correct = 0
    total_test = 0
    
    # testing aux_model
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
            outputs = aux_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            running_correct += (predicted == labels).sum().item()
            total_test += labels.size(0)

    epoch_test_acc = running_correct / total_test

    print(f'epoch {epoch} for direction searching. '
            f'aux train Acc: {epoch_train_acc:.2f}, '
            f'aux test Acc: {epoch_test_acc:.2f}')
        
print("finished small-init model direction searching.")
"""
------------------------- apply found direction --------------------------
"""
if direction_searching_method == 'lrds':
    delta_w1 = aux_model.layer1.weight.data.clone().detach() - aux_model.W1
    delta_w2 = aux_model.layer2.weight.data.clone().detach() - aux_model.W2
    delta_w1 *= scale_gap
    delta_w2 *= scale_gap
    delta_w1 = low_rank_approximation(delta_w1, delta_rank)
    delta_w2 = low_rank_approximation(delta_w2, delta_rank)
    model.layer1.weight.data += delta_w1
    model.layer2.weight.data += delta_w2
elif direction_searching_method == 'srds':
    model.load_state_dict(aux_model.state_dict())
    model.layer1.weight.data *= scale_gap
    model.layer2.weight.data *= scale_gap
elif direction_searching_method == 'cbm':
    aux_model.apply_change_based_mask(model, top_k_percent=top_k_percent, amp_factor = amp_factor)
elif direction_searching_method == 'mbm':
    aux_model.apply_magnitude_based_mask(model, top_k_percent=top_k_percent, amp_factor = amp_factor)

"""
----------------------- print the initial accuracy of main before training --------------------------------
"""
running_correct = 0
total_train = 0
with torch.no_grad():
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        running_correct += (predicted == labels).sum().item()
        total_train += labels.size(0)
init_acc = running_correct / total_train

model.eval()
running_correct = 0
total_test = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        running_correct += (predicted == labels).sum().item()
        total_test += labels.size(0)
init_test_acc = running_correct / total_test
print(f'main at initialization, before main training loop: '
        f'Train Acc: {init_acc:.2f}, '
        f'Test Acc: {init_test_acc:.2f}')
"""
------------------------------------------ main loop ---------------------------------------
"""
grads = None
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total_train = 0
    """
    jacobian_norm = 0
    ntk_norm = 0
    layer1_rank_pre = model.layer1.weight.clone().detach()
    layer2_rank_pre = model.layer2.weight.clone().detach()
    jacobian_of_each_batch = []
    """
    # optimizer.lr = scheduler.step()
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
        # compute the jacobian and ntk
        # jacobian_start = compute_jacobian(model, device, inputs)

        # optimization 
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
        elif args.filter == "smoother":
            grads = smoother(model, grads=grads, beta=args.beta, pp=args.pp)
        elif args.filter == "kalman":
            grads = gradfilter_kalman(model, grads=grads, process_noise=args.process_noise, measurement_noise=args.measurement_noise, lamb=args.lamb)
        else:
            raise ValueError(f"Invalid update filter type `{args.filter}`")

        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        running_correct += (predicted == labels).sum().item()
        total_train += labels.size(0)
        """
        # jacobian again, for the relative change
        jacobian_end = compute_jacobian(model, device, inputs)
        jacobian_change = torch.norm(jacobian_end - jacobian_start) 
        jacobian_norm += jacobian_change / torch.norm(jacobian_start) * inputs.size(0)
        jacobian_of_each_batch.append(jacobian_end)
        """
    # log the model parameters' update rank
    """
    layer1_update = model.layer1.weight.clone().detach() - layer1_rank_pre
    layer2_update = model.layer2.weight.clone().detach() - layer2_rank_pre
    layer1_update_from_init = model.layer1.weight.clone().detach() - model.W1
    layer2_update_from_init = model.layer2.weight.clone().detach() - model.W2

    update_ranks_layer1.append(compute_norm_effective_rank(layer1_update))
    update_ranks_layer2.append(compute_norm_effective_rank(layer2_update))
    update_ranks_from_init_layer1.append(compute_norm_effective_rank(layer1_update_from_init))
    update_ranks_from_init_layer2.append(compute_norm_effective_rank(layer2_update_from_init))

    jacobian_norms.append(jacobian_norm.item() / total_train)
    print(f'Epoch {epoch+1}, Norm of Jacobian Change: {jacobian_norm.item() / total_train}')
    epoch_jacobian = torch.vstack(jacobian_of_each_batch).to(device)
    emp_ntk = compute_ntk_batch(device, epoch_jacobian)
    emp_ntk_copy.append(emp_ntk)

    # print(f"the shape of the emp_ntk is {emp_ntk.shape}\n")
    if epoch == 0:
        init_emp_ntk = emp_ntk

    emp_ntk_change = torch.norm(emp_ntk - init_emp_ntk, p = 'fro').item()
    emp_ntk_similarity = compute_cosine_similarity(emp_ntk, init_emp_ntk)
    emp_ntk_similarity_log.append(emp_ntk_similarity)
    emp_ntks.append(emp_ntk_change)
    print(f'Epoch {epoch+1}, emprical NTK change: {emp_ntk_change}')
    """
    
    # Compute and log effective ranks of two layers
    layer1_effective_rank = compute_norm_effective_rank(model.layer1.weight)
    layer2_effective_rank = compute_norm_effective_rank(model.layer2.weight)
    layer1_effective_ranks.append(layer1_effective_rank)
    layer2_effective_ranks.append(layer2_effective_rank)
    layer1_norm = torch.linalg.norm(model.layer1.weight, ord = 2).item() / layer1_spec_norm_init
    layer2_norm = torch.linalg.norm(model.layer2.weight, ord = 2).item() / layer2_spec_norm_init

    layer1_spec_norm.append(layer1_norm)
    layer2_spec_norm.append(layer2_norm)


    # log the first few significant singular values of the second layer
    with torch.no_grad():
        weight_matrix = model.layer2.weight.clone().detach()
        u, s, v = torch.svd(weight_matrix)
        singular_values[:, epoch] = s[:num_singular_values].cpu().numpy()


    epoch_loss = running_loss / total_train
    epoch_acc = running_correct / total_train
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total_test = 0
    
    # testing
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            running_correct += (predicted == labels).sum().item()
            total_test += labels.size(0)
    
    epoch_loss = running_loss / total_test
    epoch_acc = running_correct / total_test
    test_loss.append(epoch_loss)
    test_acc.append(epoch_acc)
    
    print(f'Epoch {epoch + 1}/{num_epochs}, '
          f'Train Loss: {train_loss[-1]:.4f}, Train Acc: {train_acc[-1]:.2f}, '
          f'Test Loss: {test_loss[-1]:.4f}, Test Acc: {test_acc[-1]:.2f}')

    # Plotting
    epochs_plotting = range(1, len(train_acc) + 1)

    plt.plot(epochs_plotting, train_loss, label='Train Loss')
    plt.plot(epochs_plotting, test_loss, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xscale("log", base=10)
    plt.legend()
    plt.grid()
    plt.title('Loss vs Epochs')
    plt.savefig(f"results_twin_mlp/loss_{args.label}.png")
    plt.close()

    """
    # Plot emprical NTK 
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_xscale('log', base = 10)  # Set x-axis to log scale
    ax1.set_ylabel('Accuracy (%)', color=color)
    ax1.plot(train_acc, label='Train Accuracy', color=color)
    ax1.plot(test_acc, label='Test Accuracy', linestyle='dashed', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

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


    # Plot update rank changes
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_xscale('log', base = 10)  # Set x-axis to log scale
    ax1.set_ylabel('Accuracy (%)', color=color)
    ax1.plot(train_acc, label='Train Accuracy', color=color)
    ax1.plot(test_acc, label='Test Accuracy', linestyle='dashed', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('weights update rank', color=color)
    ax2.plot(update_ranks_layer1, label='layer1', color=color)
    ax2.plot(update_ranks_layer2, label='layer2',linestyle='dashed' ,color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Add legends
    fig.tight_layout()  # Ensure the right y-label is not slightly clipped
    fig.legend(loc='upper left')

    plt.title('Accuracy and weights update rank vs Epochs')
    plt.savefig(f"results_twin_mlp/weights_update_rank_{args.label}.png")
    plt.close()

    # Plot update rank changes from the init
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_xscale('log', base = 10)  # Set x-axis to log scale
    ax1.set_ylabel('Accuracy (%)', color=color)
    ax1.plot(train_acc, label='Train Accuracy', color=color)
    ax1.plot(test_acc, label='Test Accuracy', linestyle='dashed', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('weights update rank from init', color=color)
    ax2.plot(update_ranks_from_init_layer1, label='layer1', color=color)
    ax2.plot(update_ranks_from_init_layer2, label='layer2', linestyle='dashed', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Add legends
    fig.tight_layout()  # Ensure the right y-label is not slightly clipped
    fig.legend(loc='upper left')

    plt.title('Accuracy and weights update rank vs Epochs')
    plt.savefig(f"results_twin_mlp/weights_update_rank_from_init_{args.label}.png")
    plt.close()
    """
    
    # Plot effective ranks
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_xscale('log', base = 10)  # Set x-axis to log scale
    ax1.set_ylabel('Accuracy (%)', color=color)
    ax1.plot(epochs_plotting, train_acc, label='Train Accuracy', color=color)
    ax1.plot(epochs_plotting, test_acc, label='Test Accuracy', linestyle='dashed', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('effective ranks', color=color)
    ax2.plot(epochs_plotting, layer1_effective_ranks, label='layer 1', color=color)
    ax2.plot(epochs_plotting, layer2_effective_ranks, label='layer 2', linestyle='dashed', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Add legends
    fig.tight_layout()  # Ensure the right y-label is not slightly clipped
    fig.legend(loc='upper left')

    plt.title('Accuracy and Weight Ranks vs Epochs')
    plt.savefig(f"results_twin_mlp/rank_{args.label}.png")
    plt.close()

    # Plot effective ranks
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_xscale('log', base = 10)  # Set x-axis to log scale
    ax1.set_ylabel('Accuracy (%)', color=color)
    ax1.plot(epochs_plotting, train_acc, label='Train Accuracy', color=color)
    ax1.plot(epochs_plotting, test_acc, label='Test Accuracy', linestyle='dashed', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a second y-axis for the Jacobian norms
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('spectral norm', color=color)
    ax2.plot(epochs_plotting, layer1_spec_norm, label='layer 1', color=color)
    ax2.plot(epochs_plotting, layer2_spec_norm, label='layer 2', linestyle='dashed', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Add legends
    fig.tight_layout()  # Ensure the right y-label is not slightly clipped
    fig.legend(loc='upper left')

    plt.title('Accuracy and Spectral Norm vs Epochs')
    plt.savefig(f"results_twin_mlp/spec_norm_{args.label}.png")
    plt.close()


    # Define a colormap
    cmap = plt.get_cmap('tab10')  # 'tab10' has 10 different colors

    # Plot the singular values evolution
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_xscale('log', base=10)  # Set x-axis to log scale
    ax1.set_ylabel('Accuracy (%)', color=color)
    ax1.plot(epochs_plotting, train_acc, label='Train Accuracy', color=color)
    ax1.plot(epochs_plotting, test_acc, label='Test Accuracy', linestyle='dashed', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a second y-axis for the singular values
    ax2 = ax1.twinx()
    ax2.set_ylabel('S.V. ', color='tab:green')

    # Plot each singular value with a different color
    for i in range(num_singular_values):
        ax2.plot(singular_values[i, :], label=f'Singular Value {i+1}', color=cmap(i + 1))
    ax2.tick_params(axis='y', labelcolor='tab:green')

    # Add legends
    fig.tight_layout()  # Ensure the right y-label is not slightly clipped
    fig.legend(loc='upper left')

    plt.title('Accuracy and Singular Values vs Epochs')
    plt.savefig(f"results_twin_mlp/singular_val_{args.label}.png")
    plt.close()

"""
final_emp_ntk = emp_ntk

emp_ntk_alignment = []
for epoch in range(num_epochs):
    emp_ntk_alignment.append(compute_cosine_similarity(emp_ntk_copy[epoch], final_emp_ntk))


# Plot NTK alignment

fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Epoch')
ax1.set_xscale('log', base = 10)  # Set x-axis to log scale
ax1.set_ylabel('Accuracy (%)', color=color)
ax1.plot(train_acc, label='Train Accuracy', color=color)
ax1.plot(test_acc, label='Test Accuracy', linestyle='dashed', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel('NTK alignment', color=color)
ax2.plot(emp_ntk_similarity_log, label='alignment with init', color=color)
ax2.plot(emp_ntk_alignment, label='alignment with final', linestyle='dashed', color=color)
ax2.tick_params(axis='y', labelcolor=color)

# Add legends
fig.tight_layout()  # Ensure the right y-label is not slightly clipped
fig.legend(loc='upper left')

plt.title('Accuracy and NTK alignment vs Epochs')
plt.savefig(f"results_twin_mlp/NTK_alignment_{args.label}.png")
plt.close()
"""