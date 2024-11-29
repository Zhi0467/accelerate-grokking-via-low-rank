from PIL import Image
import math
from argparse import ArgumentParser
from itertools import permutations
import copy
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F

from grokfast import *
from model import *
from model import compute_jacobian
from optimizers import *
from arg_parser import *
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

def compute_sparsity(model):
    total_params = 0
    zero_params = 0
    for param in model.parameters():
        total_params += param.numel()
        zero_params += (param == 0).sum().item()
    sparsity = zero_params / total_params
    return sparsity

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

def extract_weight_matrices(layer):
    attn_weight = layer.attn.in_proj_weight 
    output_weight = layer.attn.out_proj.weight  # W_O
    ffn_weight_1 = layer.mlp[0].weight  # W_1
    ffn_weight_2 = layer.mlp[2].weight  # W_2

    return {
        'attn': attn_weight,
        'output': output_weight,
        'ffn1': ffn_weight_1,
        'ffn2': ffn_weight_2,
    }

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

def compute_norm_shannon_entropy(weight_matrix):
    abs_weights = torch.abs(weight_matrix)
    l1_norm = torch.sum(abs_weights)
    abs_weights = abs_weights / l1_norm
    log_abs_weights = torch.log(abs_weights)
    entropy = -torch.sum(abs_weights * log_abs_weights).item()
    return entropy

def plot_attention_maps(attention_maps, epoch):
    saved_filenames = []
    for layer_idx, layer_attentions in enumerate(attention_maps):
        if layer_attentions is not None:
            print("saving attention matrices.")
            # Average across the batch dimension
            avg_attention = layer_attentions.mean(dim=0)  # Shape will be (L, S) after averaging
            plt.figure(figsize=(6, 6))
            sns.heatmap(avg_attention.detach().cpu().numpy(), cmap="coolwarm")
            plt.title(f'Optimization step {epoch} - Layer {layer_idx + 1} (averaged over batch)')
            plt.xlabel('Input Sequence Position')
            plt.ylabel('Input Sequence Position')
            filename = f"results_transformer/Heatmap/Optimization-step-{epoch}-Layer-{layer_idx + 1}.png"
            plt.savefig(filename)
            saved_filenames.append(filename)
            plt.close()  # Close the figure to save memory
    return saved_filenames
    
def concatenate_images(filenames, output_filename):
    images = [Image.open(f) for f in filenames]

    # Assuming all images are the same size, get dimensions
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    # Create a blank canvas to paste the images
    new_image = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_image.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    new_image.save(output_filename)


def gradient_quotient(loss, params, eps=1e-5):
    """
    this function computes the gradient quotient of a specific loss given params
    """
    grad = torch.autograd.grad(loss, params, retain_graph=True, create_graph=True)
    prod = torch.autograd.grad(sum([(g**2).sum() / 2 for g in grad]), params, retain_graph=True, create_graph=True)
    out = sum([((g - p) / (g + eps * (2*(g >= 0).float() - 1).detach()) - 1).abs().sum() for g, p in zip(grad, prod)])
    return out / sum([p.data.nelement() for p in params])

def compute_initial_gradient_quotient(model, criterion, train_loader, device):
    """
    Computes the initial gradient quotient for the provided model on the first batch of the training data.
    
    Parameters:
    - model (nn.Module): The neural network model (SimpleMLP) to evaluate.
    - criterion (nn.Module): Loss function, e.g., CrossEntropyLoss.
    - train_loader (DataLoader): DataLoader object containing the training data.
    - device (torch.device): Device to run the computation on ('cpu' or 'cuda').
    
    Returns:
    - float: The computed gradient quotient.
    """
    model.to(device)
    model.train()  # Ensure the model is in training mode
    
    total_gq = 0.0
    num_batches = 0

    # Iterate over all batches in the training data
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # Forward pass to compute the loss
        output = model(X_batch)
        loss = criterion(output, y_batch)
        
        # Compute the gradient quotient for this batch
        gq_value = gradient_quotient(loss, list(model.parameters()))
        
        # Accumulate the total gradient quotient and batch count
        total_gq += gq_value.item()
        num_batches += 1

    # Return the average gradient quotient
    average_gq = total_gq / num_batches if num_batches > 0 else 0.0
    return average_gq

def compute_average_gq_multiple_runs_on_fixed_model(model, criterion, train_loader, device, num_runs=5):
    total_gq = 0.0
    model = model.to(device)
    for _ in range(num_runs):
        # Compute the average gradient quotient for the current run
        avg_gq = compute_initial_gradient_quotient(model, criterion, train_loader, device)
        total_gq += avg_gq

    # Compute the mean gradient quotient across all runs
    mean_gq = total_gq / num_runs if num_runs > 0 else 0.0
    return mean_gq

def compute_average_gq_multiple_runs(model_class, criterion, train_loader, device, num_runs=5, **model_kwargs):
    """
    Computes the average gradient quotient over multiple runs.
    
    Parameters:
    - model_class (nn.Module): The neural network class (e.g., SimpleMLP) to instantiate.
    - criterion (nn.Module): Loss function, e.g., CrossEntropyLoss.
    - train_loader (DataLoader): DataLoader object containing the training data.
    - device (torch.device): Device to run the computation on ('cpu' or 'cuda').
    - num_runs (int): Number of runs to average the gradient quotient over.
    - model_kwargs (dict): Additional keyword arguments for the model initialization.
    
    Returns:
    - float: The averaged gradient quotient over the specified number of runs.
    """
    total_gq = 0.0

    for _ in range(num_runs):
        # Instantiate a new model for each run
        model = model_class(**model_kwargs).to(device)
        
        # Compute the average gradient quotient for the current run
        avg_gq = compute_initial_gradient_quotient(model, criterion, train_loader, device)
        total_gq += avg_gq

    # Compute the mean gradient quotient across all runs
    mean_gq = total_gq / num_runs if num_runs > 0 else 0.0
    return mean_gq

def metainit_with_dataset(model, criterion, train_loader, device, lr=0.1, momentum=0.9, steps=500, eps=1e-5):
    """
    Meta-initialization procedure tailored for a real dataset.
    
    Parameters:
    - model (nn.Module): The neural network model to initialize.
    - criterion (nn.Module): Loss function, e.g., CrossEntropyLoss.
    - train_loader (DataLoader): DataLoader object containing the training data.
    - lr (float): Learning rate for the meta-initialization.
    - momentum (float): Momentum for updating parameter norms.
    - steps (int): Number of steps to perform the meta-initialization.
    - eps (float): Small value to avoid division by zero in gradient quotient.
    
    Returns:
    - None: Adjusts the model's parameters in place.
    """
    model.eval()  # Switch to evaluation mode
    
    # Filter parameters that require gradient and are not biases or single weights
    params = [p for p in model.parameters() if p.requires_grad and len(p.size()) >= 2]
    memory = [0] * len(params)  # Initialize momentum memory

    # Iterate for the given number of steps
    for i in range(steps):
        # Iterate over batches from the dataset
        for X_batch, y_batch in train_loader:
            # Move input data to the GPU
            input = X_batch.to(device)
            target = y_batch.to(device)

            # Forward pass to compute the loss
            loss = criterion(model(input), target)
            
            # Compute the gradient quotient for this batch
            gq = gradient_quotient(loss, list(model.parameters()), eps)
            
            # Compute gradients with respect to the gradient quotient
            grad = torch.autograd.grad(gq, params)
            
            # Update parameters' norms based on the gradients
            for j, (p, g_all) in enumerate(zip(params, grad)):
                norm = p.data.norm().item()
                g = torch.sign((p.data * g_all).sum() / norm)
                memory[j] = momentum * memory[j] - lr * g.item()
                new_norm = norm + memory[j]
                p.data.mul_(new_norm / norm)

        # check the current average gq
        current_gq = compute_initial_gradient_quotient(model, criterion, train_loader, device)
        # Print gradient quotient for monitoring
        print(f"Step:{i} GQ = {current_gq}")
    print("Finished Meta Learning.")

def print_model_rank(model):
    for name, param in model.named_parameters():
        if param.requires_grad and len(param.size()) >= 2:
            rank = torch.linalg.matrix_rank(param.data).item()
            print(f"Layer {name}: Rank = {rank}")


def request_for_gq(models, filters, names, train_loader, test_loader, criterion, optimizer_class, lr, num_epochs, device, num_gq_runs=5, wd = 0.0):
    """
    Train a list of models for a given number of epochs and compute the mean gradient quotient (GQ) after each epoch.
    
    Parameters:
        models (list): List of PyTorch models to be trained.
        filters (list): List of filters (e.g., "none" or "ema") for each model.
        train_loader (DataLoader): DataLoader for the training dataset.
        test_loader (DataLoader): DataLoader for the testing dataset.
        criterion: Loss function (e.g., nn.CrossEntropyLoss()).
        optimizer_class: PyTorch optimizer class (e.g., torch.optim.SGD, torch.optim.Adam).
        lr (float): Learning rate.
        num_epochs (int): Number of training epochs.
        device: Device to use ('cuda' or 'cpu').
        num_gq_runs (int): Number of runs to compute the mean gradient quotient (GQ).
    
    Returns:
        results (dict): Dictionary containing training and test loss/accuracy per epoch for each model and mean GQ per epoch.
    """

    # Initialize a dictionary to store results for each model
    results = {
        'train_loss': [[] for _ in range(len(models))],
        'test_loss': [[] for _ in range(len(models))],
        'train_acc': [[] for _ in range(len(models))],
        'test_acc': [[] for _ in range(len(models))],
        'mean_gq': [[] for _ in range(len(models))]  # List of mean GQ values per epoch for each model
    }
    
    # Create separate optimizers for each model
    optimizers = [optimizer_class(model.parameters(), lr=lr, weight_decay = wd) for model in models]
    
    # Move each model to the appropriate device
    for model in models:
        model.to(device)

    grads = None
    for epoch in range(num_epochs):
        for idx, (model, optimizer) in enumerate(zip(models, optimizers)):
            model.train()  # Set the model to training mode
            running_loss = 0.0
            running_correct = 0
            total_train = 0

            # Training loop
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                if filters[idx] == "none":
                    pass
                elif filters[idx] == "ema":
                    grads = gradfilter_ema(model, grads=grads, alpha=0.98, lamb=2.0)
                else:
                    raise ValueError(f"Invalid update filter type {filters[idx]}") 
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                predicted = (outputs >= 0).float() if criterion.__class__.__name__ == "BCELoss" else torch.argmax(outputs, dim=1)
                running_correct += (predicted == labels).sum().item()
                total_train += labels.size(0)

            # Calculate average loss and accuracy for training
            train_loss = running_loss / total_train
            train_acc = running_correct / total_train

            # Store training results
            results['train_loss'][idx].append(train_loss)
            results['train_acc'][idx].append(train_acc)

            # Evaluate on the test set
            model.eval()  # Set the model to evaluation mode
            running_loss = 0.0
            running_correct = 0
            total_test = 0

            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item() * inputs.size(0)
                    predicted = (outputs >= 0).float() if criterion.__class__.__name__ == "BCELoss" else torch.argmax(outputs, dim=1)
                    running_correct += (predicted == labels).sum().item()
                    total_test += labels.size(0)

            # Calculate average loss and accuracy for testing
            test_loss = running_loss / total_test
            test_acc = running_correct / total_test

            # Store test results
            results['test_loss'][idx].append(test_loss)
            results['test_acc'][idx].append(test_acc)

            # Compute mean GQ after each epoch for the current model
            mean_gq = compute_average_gq_multiple_runs_on_fixed_model(model, criterion, train_loader, device, num_runs=num_gq_runs)
            results['mean_gq'][idx].append(mean_gq)
            print(f'Model {idx + 1}, Epoch {epoch + 1}/{num_epochs}, '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}, '
                  f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}, '
                  f'Mean GQ: {mean_gq:.4f}')
    
    # Plot the results for each model
    fig, axs = plt.subplots(2, 1, figsize=(12, 18))
    epochs = range(1, num_epochs + 1)

    # Plot training and testing accuracy
    for idx in range(len(models)):
        label = names[idx]  # Ensure label is a string
        print(label)
        color = plt.cm.tab10(idx)  # Use color map for different colors
        # Train and test accuracy subplot
        axs[0].plot(epochs, results['train_acc'][idx], label=label + ' Train Acc', color=color)
        axs[0].plot(epochs, results['test_acc'][idx], linestyle='--', label=label + ' Test Acc', color=color)
    
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_title('Training and Testing Accuracy over Epochs')
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_xscale('log')

    # Plot mean GQ progression
    for idx in range(len(models)):
        label = names[idx]
        axs[1].plot(epochs, results['mean_gq'][idx], label=label + ' Mean GQ')

    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Mean Gradient Quotient (GQ)')
    axs[1].set_title('Mean GQ Progression over Epochs for Each Model')
    axs[1].legend()
    axs[1].grid(True)
    plt.xscale('log')
    plt.show()

    return results

def request_for_gq_regression(models, filters, names, train_loader, test_loader, criterion, optimizer_class, lr, num_epochs, device, num_gq_runs=5, wd=0.0):
    """
    Train a list of models for a given number of epochs and compute the mean gradient quotient (GQ) after each epoch.
    Loss is normalized by the initial loss to compare models with different initialization scales.
    
    Parameters:
        models (list): List of PyTorch models to be trained.
        filters (list): List of filters (e.g., "none" or "ema") for each model.
        names (list): List of names for each model (for plotting purposes).
        train_loader (DataLoader): DataLoader for the training dataset.
        test_loader (DataLoader): DataLoader for the testing dataset.
        criterion: Loss function (e.g., nn.MSELoss()).
        optimizer_class: PyTorch optimizer class (e.g., torch.optim.SGD, torch.optim.Adam).
        lr (float): Learning rate.
        num_epochs (int): Number of training epochs.
        device: Device to use ('cuda' or 'cpu').
        num_gq_runs (int): Number of runs to compute the mean gradient quotient (GQ).
        wd (float): Weight decay for the optimizer.
    
    Returns:
        results (dict): Dictionary containing training and test loss/accuracy per epoch for each model and mean GQ per epoch.
    """

    # Initialize a dictionary to store results for each model
    results = {
        'train_loss': [[] for _ in range(len(models))],
        'test_loss': [[] for _ in range(len(models))],
        'mean_gq': [[] for _ in range(len(models))]  # List of mean GQ values per epoch for each model
    }
    
    # Create separate optimizers for each model
    optimizers = [optimizer_class(model.parameters(), lr=lr, weight_decay=wd) for model in models]
    
    # Move each model to the appropriate device
    for model in models:
        model.to(device)
    
    # Compute initial losses for normalization
    initial_losses = []
    for model in models:
        model.eval()  # Set the model to evaluation mode
        total_initial_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_initial_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
        
        initial_loss = total_initial_loss / total_samples if total_samples > 0 else 1.0
        initial_losses.append(initial_loss)

    grads = None
    for epoch in range(num_epochs):
        for idx, (model, optimizer) in enumerate(zip(models, optimizers)):
            model.train()  # Set the model to training mode
            running_loss = 0.0
            total_train = 0

            # Training loop
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Apply gradient filtering if necessary
                if filters[idx] == "none":
                    pass
                elif filters[idx] == "ema":
                    grads = gradfilter_ema(model, grads=grads, alpha=0.98, lamb=2.0)
                else:
                    raise ValueError(f"Invalid update filter type {filters[idx]}") 
                
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                total_train += labels.size(0)

            # Calculate average loss and normalized accuracy for training
            train_loss = running_loss / total_train
            train_loss_normalized = train_loss / initial_losses[idx] if initial_losses[idx] != 0 else train_loss

            # Store training results
            results['train_loss'][idx].append(train_loss_normalized)

            # Evaluate on the test set
            model.eval()  # Set the model to evaluation mode
            running_loss = 0.0
            running_correct = 0
            total_test = 0

            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item() * inputs.size(0)
                    total_test += labels.size(0)

            # Calculate average loss and accuracy for testing
            test_loss = running_loss / total_test
            test_loss_normalized = test_loss / initial_losses[idx] if initial_losses[idx] != 0 else test_loss

            # Store test results
            results['test_loss'][idx].append(test_loss_normalized)

            # Compute mean GQ after each epoch for the current model
            mean_gq = compute_average_gq_multiple_runs_on_fixed_model(model, criterion, train_loader, device, num_runs=num_gq_runs)
            results['mean_gq'][idx].append(mean_gq)
            print(f'Model {idx + 1}, Epoch {epoch + 1}/{num_epochs}, '
                  f'Train Loss (Norm): {train_loss_normalized:.4f}, '
                  f'Test Loss (Norm): {test_loss_normalized:.4f}, '
                  f'Mean GQ: {mean_gq:.4f}')
    
    # Plot the results for each model
    fig, axs = plt.subplots(2, 1, figsize=(12, 18))
    epochs = range(1, num_epochs + 1)

    # Plot training and testing accuracy
    for idx in range(len(models)):
        label = names[idx]  # Ensure label is a string
        color = plt.cm.tab10(idx)  # Use color map for different colors
        # Train and test accuracy subplot
        axs[0].plot(epochs, results['train_loss'][idx], label=label + ' Train Loss', color=color)
        axs[0].plot(epochs, results['test_loss'][idx], linestyle='--', label=label + ' Test Loss', color=color)
    
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Normalized Loss (by initial loss)')
    axs[0].set_title('Normalized Loss over Epochs')
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_xscale('log')

    # Plot mean GQ progression
    for idx in range(len(models)):
        label = names[idx]
        axs[1].plot(epochs, results['mean_gq'][idx], label=label + ' Mean GQ')

    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Mean Gradient Quotient (GQ)')
    axs[1].set_title('Mean GQ Progression over Epochs for Each Model')
    axs[1].legend()
    axs[1].grid(True)
    plt.xscale('log')
    plt.show()

    return results

def generate_gaussian_regression_data(num_samples, input_dim, noise_std=0.1):
    """
    Generate synthetic Gaussian data for regression with a known ground truth.
    
    Parameters:
        num_samples (int): Number of samples to generate.
        input_dim (int): Dimensionality of the input data.
        noise_std (float): Standard deviation of the noise added to the outputs.
    
    Returns:
        X (torch.Tensor): Input features of shape (num_samples, input_dim).
        y (torch.Tensor): Targets of shape (num_samples, 1).
        true_weights (torch.Tensor): Ground truth weight vector.
    """
    # Random Gaussian input data
    X = torch.randn(num_samples, input_dim)
    
    # Ground truth weights
    true_weights = torch.randn(input_dim, 1)
    
    # Generate targets with Gaussian noise
    y = X @ true_weights + noise_std * torch.randn(num_samples, 1)
    
    return X, y, true_weights

def generate_polynomial_regression_data(num_samples, input_dim, noise_std=0.1, degree=3):
    """
    Generate synthetic polynomial regression data with a known polynomial ground truth function.
    
    Parameters:
        num_samples (int): Number of samples to generate.
        input_dim (int): Dimensionality of the input data.
        noise_std (float): Standard deviation of the noise added to the outputs.
        degree (int): Degree of the polynomial used in the ground truth function.
    
    Returns:
        X (torch.Tensor): Input features of shape (num_samples, input_dim).
        y (torch.Tensor): Targets of shape (num_samples, 1).
        true_coeffs (list): List of coefficients used in the polynomial ground truth function.
    """
    # Random input data
    X = torch.randn(num_samples, input_dim)
    
    # Generate polynomial ground truth function coefficients (random for each degree)
    true_coeffs = [torch.randn(1) for _ in range(degree + 1)]  # Coefficients for terms up to degree
    y_true = torch.zeros(num_samples, 1)
    
    # Compute polynomial function: y = c0 + c1*x + c2*x^2 + ... + c_degree*x^degree
    for d in range(degree + 1):
        y_true += true_coeffs[d] * (X ** (d + 1)).sum(dim=1, keepdim=True)

    # Add noise to the target
    y = y_true + noise_std * torch.randn(num_samples, 1)
    
    return X, y, true_coeffs

def generate_nn_regression_data(num_samples, input_dim, noise_std=0.1, hidden_dim=2):
    """
    Generate synthetic regression data using a small randomly initialized neural network.
    
    Parameters:
        num_samples (int): Number of samples to generate.
        input_dim (int): Dimensionality of the input data.
        noise_std (float): Standard deviation of the noise added to the outputs.
        hidden_dim (int): Number of neurons in the hidden layer.
    
    Returns:
        X (torch.Tensor): Input features of shape (num_samples, input_dim).
        y (torch.Tensor): Targets of shape (num_samples, 1).
        model (nn.Module): The randomly initialized neural network used to generate the data.
    """
    # Define a small neural network with a single hidden layer
    class SmallNN(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super(SmallNN, self).__init__()
            self.hidden_layer = nn.Linear(input_dim, hidden_dim)
            self.output_layer = nn.Linear(hidden_dim, 1)
            self.activation = nn.ReLU()  # Non-linear activation function

        def forward(self, x):
            x = self.activation(self.hidden_layer(x))
            return self.output_layer(x)

    # Instantiate the model and freeze its parameters (to use it as a generator)
    model = SmallNN(input_dim, hidden_dim)
    model.eval()  # Set the model to evaluation mode
    
    # Generate random input data
    X = torch.randn(num_samples, input_dim)
    
    # Generate output using the network
    with torch.no_grad():
        y_true = model(X)
    
    # Add noise to the output
    y = y_true + noise_std * torch.randn_like(y_true)
    
    return X, y, model