# modular arithmetic in a 2-layer MLP
# = high dimensional sparse classification 

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
sparse_init = args.sparse_init
sparsity = args.sparsity
low_rank_switch = args.low_rank_switch
switch_to_rank = args.switch_to_rank
update_rank_percentage = args.update_rank_percentage


# Generate data and split into training and test sets
X, y = generate_data(p, 'add')
dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
train_size = int(alpha * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model, loss function, and optimizer
model = SimpleMLP(input_dim, hidden_dim, output_dim, scale = scale, rank = rank, sparse_init = sparse_init, sparsity = sparsity).to(device)
criterion = nn.CrossEntropyLoss()
if args.optimizer == 'AdamW' or args.optimizer == 'Adam':
    optimizer = getattr(torch.optim, args.optimizer)(
            model.parameters(),
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
    
# scheduler = LrScheduler(large_lr=args.lr, regular_lr=args.lr, warmup_steps = 10, cutoff_steps=args.cutoff_steps)

num_singular_values = 8

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
save_epochs = [0, 10, 100, 200, 500, 1000, num_epochs - 1]

"""
update_ranks_layer1 = []
update_ranks_layer2 = []
update_ranks_from_init_layer1 = []
update_ranks_from_init_layer2 = []
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

    if epoch == args.switch_epoch and args.low_rank_switch:
        model.initialize_low_rank(switch_to_rank)
        print(f"SWITCHING WEIGHTS TO RANK {switch_to_rank}.")

    if epoch in save_epochs:
        model.save_nfm(epoch)
    
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

        # perform low-rank projection on grads
        if args.enable_lr_update:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None and len(param.grad.shape) == 2:
                        grad = param.grad
                        print(f"{name}: pre-SVD: rank = {torch.linalg.matrix_rank(grad)}")
                        max_rank = max(grad.shape)
                        rank = int(update_rank_percentage * max_rank)
                        param.grad = low_rank_approximation(grad, rank)
                        print(f"{name}: post-SVD: rank = {torch.linalg.matrix_rank(param.grad)}")
        
         
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        running_correct += (predicted == labels).sum().item()
        total_train += labels.size(0)
        """
        jacobian_end = compute_jacobian(model, device, inputs)
        jacobian_change = torch.norm(jacobian_end - jacobian_start) 
        jacobian_norm += jacobian_change / torch.norm(jacobian_start) * inputs.size(0)
        jacobian_of_each_batch.append(jacobian_end)
        """

        # ntk
        """
        ntk_end = compute_ntk_batch(device, inputs, jacobian_end_wrt_inputs)
        ntk_change = torch.norm(ntk_end - ntk_start)
        ntk_norm += ntk_change / torch.norm(ntk_start) * inputs.size(0)
        """
    """
    # log the model parameters' update rank
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
    plt.savefig(f"results_mlp/loss_{args.label}.png")
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
    plt.savefig(f"results_mlp/emprical_ntk_{args.label}.png")
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
    plt.savefig(f"results_mlp/weights_update_rank_{args.label}.png")
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
    plt.savefig(f"results_mlp/weights_update_rank_from_init_{args.label}.png")
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

    # Create a second y-axis for the Jacobian norms
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
    plt.savefig(f"results_mlp/rank_{args.label}.png")
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
    plt.savefig(f"results_mlp/spec_norm_{args.label}.png")
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
    plt.savefig(f"results_mlp/singular_val_{args.label}.png")
    plt.close()

concatenate_images(model.saved_filenames_1, f'results_mlp/NFM_layer1_{args.label}.png')
concatenate_images(model.saved_filenames_2, f'results_mlp/NFM_layer2_{args.label}.png')  

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
plt.savefig(f"results_mlp/NTK_alignment_{args.label}.png")
plt.close()
"""
