import torchvision
import torchvision.transforms as transforms
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
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = Arg_parser()
args = parser.return_args()

# Parameters
input_dim = 32 * 32 * 3
hidden_dim = args.hidden_dim
output_dim = 10
scale = args.init_scale
num_epochs = args.num_epochs
print_interval = int(num_epochs / 500)
batch_size = args.batch_size
lr = args.lr
beta = args.beta
rank = args.init_rank

# Transform CIFAR-10 images to tensors and normalize them
transform = transforms.Compose([transforms.ToTensor(),  transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]), transforms.Lambda(lambda x: x.view(-1))])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle=True, num_workers = 2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle=False, num_workers = 2)


# Model, loss function, and optimizer
model = SimpleMLP(input_dim, hidden_dim, output_dim, scale, activation='relu', beta = beta, rank = rank).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = getattr(torch.optim, args.optimizer)(
        model.parameters(),
        lr=lr,
        weight_decay=args.weight_decay,
    )
scheduler = LrScheduler(large_lr=args.lr, regular_lr=args.lr, warmup_steps = 10, cutoff_steps=args.cutoff_steps)

num_singular_values = 8

# Initialize a 2D array to store the singular values
singular_values = np.zeros((num_singular_values, num_epochs))


# Training and logging
train_loss = []
test_loss = []
train_acc = []
test_acc = []


layer1_effective_ranks = []
layer2_effective_ranks = []
layer1_spec_norm = []
layer2_spec_norm = []
layer1_spec_norm_init = torch.linalg.norm(model.layer1.weight, ord = 2).item()
layer2_spec_norm_init = torch.linalg.norm(model.layer2.weight, ord = 2).item()

update_ranks_layer1 = []
update_ranks_layer2 = []
update_ranks_from_init_layer1 = []
update_ranks_from_init_layer2 = []

layer1_rank_init = model.layer1.weight.clone().detach()
layer2_rank_init = model.layer2.weight.clone().detach()

nfm1_rank = []
nfm2_rank = []
nfm1 = []
nfm2 = []


grads = None
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total_train = 0

    layer1_rank_pre = model.layer1.weight.clone().detach()
    layer2_rank_pre = model.layer2.weight.clone().detach()


    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU

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
        
         
        optimizer.lr = scheduler.step()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        running_correct += (predicted == labels).sum().item()
        total_train += labels.size(0)

        # ntk
        """
        ntk_end = compute_ntk_batch(device, inputs, jacobian_end_wrt_inputs)
        ntk_change = torch.norm(ntk_end - ntk_start)
        ntk_norm += ntk_change / torch.norm(ntk_start) * inputs.size(0)
        """
    
    # log the model parameters' update rank
    layer1_update = model.layer1.weight.clone().detach() - layer1_rank_pre
    layer2_update = model.layer2.weight.clone().detach() - layer2_rank_pre
    layer1_update_from_init = model.layer1.weight.clone().detach() - layer1_rank_init
    layer2_update_from_init = model.layer2.weight.clone().detach() - layer2_rank_init

    update_ranks_layer1.append(compute_norm_effective_rank(layer1_update))
    update_ranks_layer2.append(compute_norm_effective_rank(layer2_update))
    update_ranks_from_init_layer1.append(compute_norm_effective_rank(layer1_update_from_init))
    update_ranks_from_init_layer2.append(compute_norm_effective_rank(layer2_update_from_init))
    

    # Compute and log effective ranks of two layers
    layer1_effective_rank = compute_norm_effective_rank(model.layer1.weight)
    layer2_effective_rank = compute_norm_effective_rank(model.layer2.weight)
    layer1_effective_ranks.append(layer1_effective_rank)
    layer2_effective_ranks.append(layer2_effective_rank)
    layer1_norm = torch.linalg.norm(model.layer1.weight, ord = 2).item() / layer1_spec_norm_init
    layer2_norm = torch.linalg.norm(model.layer2.weight, ord = 2).item() / layer2_spec_norm_init

    layer1_spec_norm.append(layer1_norm)
    layer2_spec_norm.append(layer2_norm)


    nfm1_rank.append(compute_norm_effective_rank(model.nfm1))
    nfm2_rank.append(compute_norm_effective_rank(model.nfm2))
    nfm1.append(model.nfm1)
    nfm2.append(model.nfm2)


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
    plt.savefig(f"results_mlp_cifar10/loss_{args.label}.png")
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
    plt.savefig(f"results_mlp_cifar10/weights_update_rank_{args.label}.png")
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
    plt.savefig(f"results_mlp_cifar10/weights_update_rank_from_init_{args.label}.png")
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
    ax2.plot(layer2_effective_ranks, label='layer 2', linestyle='dashed', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Add legends
    fig.tight_layout()  # Ensure the right y-label is not slightly clipped
    fig.legend(loc='upper left')

    plt.title('Accuracy and Weight Ranks vs Epochs')
    plt.savefig(f"results_mlp_cifar10/rank_{args.label}.png")
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
    ax2.set_ylabel('spectral norm', color=color)
    ax2.plot(layer1_spec_norm, label='layer 1', color=color)
    ax2.plot(layer2_spec_norm, label='layer 2', linestyle='dashed', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Add legends
    fig.tight_layout()  # Ensure the right y-label is not slightly clipped
    fig.legend(loc='upper left')

    plt.title('Accuracy and Spectral Norm vs Epochs')
    plt.savefig(f"results_mlp_cifar10/spec_norm_{args.label}.png")
    plt.close()

    # Plot effective ranks for nfms
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
    ax2.set_ylabel('NFM effective rank', color=color)
    ax2.plot(nfm1_rank, label='layer 1', color=color)
    ax2.plot(nfm2_rank, label='layer 2', linestyle='dashed', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Add legends
    fig.tight_layout()  # Ensure the right y-label is not slightly clipped
    fig.legend(loc='upper left')

    plt.title('Accuracy and NFM rank vs Epochs')
    plt.savefig(f"results_mlp_cifar10/NFM_rank_{args.label}.png")
    plt.close()


    # Define a colormap
    cmap = plt.get_cmap('tab10')  # 'tab10' has 10 different colors

    # Plot the singular values evolution
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_xscale('log', base=10)  # Set x-axis to log scale
    ax1.set_ylabel('Accuracy (%)', color=color)
    ax1.plot(train_acc, label='Train Accuracy', color=color)
    ax1.plot(test_acc, label='Test Accuracy', linestyle='dashed', color=color)
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
    plt.savefig(f"results_mlp_cifar10/singular_val_{args.label}.png")
    plt.close()

final_nfm1 = model.nfm1
final_nfm2 = model.nfm2

nfm1_alignment = []
nfm2_alignment = []
for epoch in range(num_epochs):
    nfm1_alignment.append(compute_cosine_similarity(nfm1[epoch], final_nfm1))
    nfm2_alignment.append(compute_cosine_similarity(nfm2[epoch], final_nfm2))

# Plot nfm alignment

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
ax2.set_ylabel('NFM alignment', color=color)
ax2.plot(nfm1_alignment, label='layer 1', color=color)
ax2.plot(nfm2_alignment, label='layer 2', linestyle='dashed', color=color)
ax2.tick_params(axis='y', labelcolor=color)

# Add legends
fig.tight_layout()  # Ensure the right y-label is not slightly clipped
fig.legend(loc='upper left')

plt.title('Accuracy and NFM alignment vs Epochs')
plt.savefig(f"results_mlp_cifar10/NFM_alignment_{args.label}.png")
plt.close()