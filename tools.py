from PIL import Image
import math
from argparse import ArgumentParser
from itertools import permutations
import copy
import numpy as np

import matplotlib.pyplot as plt
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
