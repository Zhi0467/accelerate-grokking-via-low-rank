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
import seaborn as sns

from grokfast import *

class LoRALinear(nn.Module):
    """Low-Rank Linear Layer for LoRA"""

    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank
        self.W = nn.Linear(in_features, out_features, bias=False)  # Original weight matrix
        # freeze it
        for param in self.W.parameters():
            param.requires_grad = False
        self.A = nn.Parameter(torch.randn(out_features, rank) / rank)  # Low-rank matrix A
        self.B = nn.Parameter(torch.randn(in_features, rank) / rank)  # Low-rank matrix B
        self.weight = self.W.weight.clone().detach() + self.A @ self.B.T

    def forward(self, x):
        self.weight = self.W.weight.clone().detach() + self.A @ self.B.T
        # First, apply the frozen weight matrix
        output = self.W(x)
        # Then, apply the LoRA low-rank transformation
        low_rank_output = x @ self.B  # shape (batch_size, rank)
        low_rank_output = low_rank_output @ self.A.T  # shape (batch_size, out_features)
        return output + low_rank_output
        
class Block(nn.Module):
    """Causal transformer block
    """

    def __init__(self, dim, num_heads, beta = None, rank = None, LoRA_rank = None, attn_freeze = True, first_block_freeze = False):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.ln_2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        # LoRA applied to the MLP layers
        self.mlp_fc1 = LoRALinear(dim, dim * 4, LoRA_rank) if LoRA_rank is not None else nn.Linear(dim, dim * 4)
        self.mlp_fc2 = LoRALinear(dim * 4, dim, LoRA_rank) if LoRA_rank is not None else nn.Linear(dim * 4, dim)
        self.activation = nn.GELU()
        self.mlp = nn.Sequential(
            self.mlp_fc1,
            self.activation,
            self.mlp_fc2,
        )
        
        self.rank = rank
        # Split the in_proj_weight into query, key, and value weight matrices
        query_weight, key_weight, value_weight = self.attn.in_proj_weight.chunk(3, dim=0)
        self.value_matrix = value_weight

        # Randomly initialized attention weights but set to not update during backprop
        if attn_freeze: 
            print("freezing the attention!")
            for param in self.attn.parameters():
                param.requires_grad = False
        else:
            print("not freezing the attention nor the layer norm!")

        if first_block_freeze:
            print("freezing the entire block!")
            for param in self.parameters():
                param.requires_grad = False
        else:
            print("not freezing the block!")
            

        if beta is not None:
            self.attn.in_proj_weight.data *= beta
            # self.attn.out_proj.weight.data *= beta

        if rank is not None:
            print(f"using rank {rank} initialization.")
            self.attn.in_proj_weight.data = self.low_rank_approximation(self.attn.in_proj_weight.data, rank)
            self.attn.out_proj.weight.data = self.low_rank_approximation(self.attn.out_proj.weight.data, rank)
            self.mlp[0].weight.data = self.low_rank_approximation(self.mlp[0].weight.data, rank)
            self.mlp[2].weight.data = self.low_rank_approximation(self.mlp[2].weight.data, rank)

        # self.sparse_mask_on_attn()



    def forward(self, x, need_attn_weights = False):
        attn_mask = torch.full(
            (len(x), len(x)), -float("Inf"), device=x.device, dtype=x.dtype
        )
        attn_mask = torch.triu(attn_mask, diagonal=1)
        attn_mask[torch.isnan(attn_mask)] = 0.0 # fixes all 'nan' on 'mps' device
        attention_matrices = 0

        x = self.ln_1(x)
        if need_attn_weights:
          a, attention_matrices = self.attn(x, x, x, attn_mask=attn_mask, need_weights=True)
        else:
          a, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = x + a
        # Apply LoRA-augmented MLP
        x = self.ln_2(x)
        m = self.mlp_fc1(x)
        m = self.activation(m)
        m = self.mlp_fc2(m)
        
        x = x + m
        return x, attention_matrices
    
    def low_rank_approximation(self, matrix, rank):
        # Perform SVD on the attention weights matrix
        U, S, V = torch.svd(matrix)
        
        # Compute the sum of all singular values (original nuclear norm)
        original_nuclear_norm = S.sum()
        
        # Retain only the top 'rank' singular values
        S_reduced = S[:rank]
        U = U[:, :rank]
        V = V[:, :rank]
        
        # Compute the sum of the reduced singular values (current nuclear norm)
        reduced_nuclear_norm = S_reduced.sum()
        
        # Scale the reduced singular values to preserve the original nuclear norm
        scaling_factor = original_nuclear_norm / reduced_nuclear_norm
        S_reduced = S_reduced * scaling_factor
        
        # Recompose the matrix with the scaled singular values
        S_reduced = torch.diag(S_reduced)
        low_rank_matrix = torch.mm(U, torch.mm(S_reduced, V.t()))
        
        return low_rank_matrix

    def sparse_mask_on_attn(self, sparsity_level = 0.9):
        # Get the number of parameters in the layer
        num_params = self.attn.in_proj_weight.data.numel()
        # Determine the number of non-zero weights
        num_nonzero = int((1 - sparsity_level) * num_params)
        
        # Create a mask with the desired sparsity level
        mask = torch.zeros(num_params)
        mask[:num_nonzero] = 1
        mask = mask[torch.randperm(num_params)].view_as(self.attn.in_proj_weight.data)
        
        # Apply the mask
        self.attn.in_proj_weight.data.mul_(mask)
        print(f"randomly sparse masked attention in_proj with sparsity {sparsity_level}.")

class Decoder(nn.Module):
    """Causal Transformer decoder
    """

    def __init__(self, dim=128, num_layers=2, num_heads=4, num_tokens=97, seq_len=5, beta = None, rank = None, LoRA_rank = None, attn_freeze = True, first_block_freeze = False):
        super().__init__()
        self.token_embeddings = nn.Embedding(num_tokens, dim)
        self.position_embeddings = nn.Embedding(seq_len, dim)
        if rank is not None:
            self.token_embeddings.weight.data = self.low_rank_approximation(self.token_embeddings.weight.data, rank)
            self.position_embeddings.weight.data = self.low_rank_approximation(self.position_embeddings.weight.data, rank)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(Block(dim, num_heads, beta, rank, LoRA_rank, attn_freeze, first_block_freeze = first_block_freeze))
            else:
                self.layers.append(Block(dim, num_heads, beta, rank, LoRA_rank, attn_freeze))

        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_tokens, bias=False)
        # Apply low-rank approximation to the head if rank is provided
        if rank is not None:
            self.head.weight.data = self.low_rank_approximation(self.head.weight.data, rank)


    def forward(self, x, need_attn_weights = False):
        h = self.token_embeddings(x)
        positions = torch.arange(x.shape[0], device=x.device).unsqueeze(-1)
        h = h + self.position_embeddings(positions).expand_as(h)

        attention_maps = []
        for layer in self.layers:
            h, attention_matrices = layer(h, need_attn_weights)
            attention_maps.append(attention_matrices)

        h = self.ln_f(h)
        logits = self.head(h)
        return logits, attention_maps

    def low_rank_approximation(self, matrix, rank):
        # Perform SVD on the attention weights matrix
        U, S, V = torch.svd(matrix)
        
        # Compute the sum of all singular values (original nuclear norm)
        original_nuclear_norm = S.sum()
        
        # Retain only the top 'rank' singular values
        S_reduced = S[:rank]
        U = U[:, :rank]
        V = V[:, :rank]
        
        # Compute the sum of the reduced singular values (current nuclear norm)
        reduced_nuclear_norm = S_reduced.sum()
        
        # Scale the reduced singular values to preserve the original nuclear norm
        scaling_factor = original_nuclear_norm / reduced_nuclear_norm
        S_reduced = S_reduced * scaling_factor
        
        # Recompose the matrix with the scaled singular values
        S_reduced = torch.diag(S_reduced)
        low_rank_matrix = torch.mm(U, torch.mm(S_reduced, V.t()))
        
        return low_rank_matrix


def multiplication_mod_p_data(p, eq_token, op_token):
    """x◦y = x/y (mod p) for 0 ≤ x < p, 0 < y < p
    """
    x = torch.arange(p)
    y = torch.arange(1, p)
    x, y = torch.cartesian_prod(x, y).T

    eq = torch.ones_like(x) * eq_token
    op = torch.ones_like(x) * op_token
    result = (x**2 + x * y + y**2) % p

    # "All of our experiments used a small transformer trained on datasets of
    # equations of the form a◦b = c, where each of “a”, “◦”, “b”, “=”, and “c”
    # is a seperate token"
    return torch.stack([x, op, y, eq, result])

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, scale=1.0, activation = 'quadratic', beta = None, rank=None, sparse_init = 'none', sparsity = 0.8):
        super(SimpleMLP, self).__init__()
        self.D = input_dim
        self.N = hidden_dim
        self.scale = scale
        self.rank = rank
        self.activation = activation

        self.layer1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.layer2 = nn.Linear(hidden_dim, output_dim, bias=False)
        self.saved_filenames_1 = []
        self.saved_filenames_2 = []
        
        self.layer1.weight.data *= scale
        self.layer2.weight.data *= scale

        if beta is not None:
            self.layer1.weight.data *= beta

        if sparse_init == 'random':
            print(f"initialized with a random sparse mask.")
            self.random_sparse_mask(sparsity=sparsity)
        elif sparse_init == 'lottery':
            pass
        else:
            pass

        if rank is not None:
            # print(f"set init rank to be {rank}.")
            self.initialize_low_rank(rank)

        # self.reduce_spectral_norm_keep_nuclear_norm(order = 50)

        self.W1 = self.layer1.weight.data.clone().detach()
        self.W2 = self.layer2.weight.data.clone().detach()

        self.nfm1 = torch.mm(self.layer1.weight.data.t(), self.layer1.weight.data)
        self.nfm2 = torch.mm(self.layer2.weight.data.t(), self.layer2.weight.data)

    def forward(self, x):
        x = self.layer1(x)
        if self.activation == 'quadratic':
            x = x**2 
        elif self.activation == 'relu':
            x = F.relu(x)
        else:
            print("warning: you are using a linear model")
        x = self.layer2(x)

        return x 

    def initialize_low_rank_layer(self, layer, rank):
        # Get the weight matrix of the layer
        weight = layer.weight.data

        # Perform SVD on the attention weights matrix
        U, S, V = torch.svd(weight)
        
        # Retain only the top 'rank' singular values
        S_reduced = S[:rank]
        U = U[:, :rank]
        V = V[:, :rank]
        
        S_reduced = S_reduced * 2.0
        
        # Recompose the matrix with the scaled singular values
        S_reduced = torch.diag(S_reduced)
        low_rank_weight = torch.mm(U, torch.mm(S_reduced, V.t()))
        
        # Set the layer's weight to the low-rank approximation
        layer.weight.data = low_rank_weight

    def spectral_norm_reduction_with_averaging(self, layer, order):
        """
        Reduces the spectral norm of a matrix using pairwise averaging of singular values,
        while preserving the nuclear norm.
        """
        # Perform SVD on the input matrix
        matrix = layer.weight.data
        U, S, V = torch.svd(matrix)
        print("Original Spectral Norm:", torch.svd(matrix).S[0])
        print("Original Nuclear Norm:", torch.svd(matrix).S.sum())
        average = 0.0
        for i in range(len(S) - 1):
            average += S[i]
        average /= len(S)
    
        # Pairwise average the singular values to reduce the largest one
        for i in range(len(S) - 1):
            S[i] = average
    
        # Recompose the matrix with the adjusted singular values
        adjusted_matrix = U @ torch.diag(S) @ V.t()

        print("Adjusted Spectral Norm (Averaging):", torch.svd(adjusted_matrix).S[0])
        print("Adjusted Nuclear Norm (Averaging):", torch.svd(adjusted_matrix).S.sum())
    
        layer.weight.data = adjusted_matrix
    
    
    def initialize_low_rank(self, rank):
        self.initialize_low_rank_layer(self.layer1, rank)
        self.initialize_low_rank_layer(self.layer2, rank)

    def reduce_spectral_norm_keep_nuclear_norm(self, order):
        self.spectral_norm_reduction_with_averaging(self.layer1, order)
        self.spectral_norm_reduction_with_averaging(self.layer2, order)
        
    
    def random_sparse_mask_layer(self, layer, sparsity_level):
        # Get the number of parameters in the layer
        num_params = layer.weight.numel()
        # Determine the number of non-zero weights
        num_nonzero = int((1 - sparsity_level) * num_params)
        
        # Create a mask with the desired sparsity level
        mask = torch.zeros(num_params)
        mask[:num_nonzero] = 1
        mask = mask[torch.randperm(num_params)].view_as(layer.weight)
        
        # Apply the mask
        layer.weight.data.mul_(mask)
    
    def random_sparse_mask(self, sparsity = 0.8):
        self.random_sparse_mask_layer(self.layer1, sparsity)
        self.random_sparse_mask_layer(self.layer2, sparsity)

    def to(self, device):
        super(SimpleMLP, self).to(device)
        self.W1 = self.W1.to(device)
        self.W2 = self.W2.to(device)
        return self
    
    def get_weight_changes(self):
        # Calculate absolute changes in weights
        W1_change = torch.abs(self.layer1.weight.data.clone().detach() - self.W1)
        W2_change = torch.abs(self.layer2.weight.data.clone().detach() - self.W2)
        return W1_change, W2_change

    def apply_change_based_mask(self, main_model, top_k_percent=10, amp_factor = 2.0):
        W1_change, W2_change = self.get_weight_changes()

        # Flatten the changes and sort by magnitude
        W1_flat = W1_change.view(-1)
        W2_flat = W2_change.view(-1)
        
        # Determine threshold based on top_k_percent
        k1 = int(top_k_percent / 100.0 * W1_flat.numel())
        k2 = int(top_k_percent / 100.0 * W2_flat.numel())
        
        threshold_W1 = torch.topk(W1_flat, k1, sorted=False).values.min()
        threshold_W2 = torch.topk(W2_flat, k2, sorted=False).values.min()
        
        # Create masks
        mask_W1 = (W1_change >= threshold_W1).float()
        mask_W2 = (W2_change >= threshold_W2).float()
        
        # Apply masks to the main model
        with torch.no_grad():
            main_model.layer1.weight.data.mul_(mask_W1)
            main_model.layer2.weight.data.mul_(mask_W2)
            main_model.layer1.weight.data *= amp_factor
            main_model.layer2.weight.data *= amp_factor

        print("Applied change-based mask to main model.")
    
    def apply_magnitude_based_mask(self, main_model, top_k_percent = 10, amp_factor = 2.0):
        W1_abs = torch.abs(self.layer1.weight.data.clone().detach())
        W2_abs = torch.abs(self.layer2.weight.data.clone().detach())

        W1_flat = W1_abs.view(-1)
        W2_flat = W2_abs.view(-1)
        
        # Determine threshold based on top_k_percent
        k1 = int(top_k_percent / 100.0 * W1_flat.numel())
        k2 = int(top_k_percent / 100.0 * W2_flat.numel())
        
        threshold_W1 = torch.topk(W1_flat, k1, sorted=False).values.min()
        threshold_W2 = torch.topk(W2_flat, k2, sorted=False).values.min()
        
        # Create masks
        mask_W1 = (W1_abs >= threshold_W1).float()
        mask_W2 = (W2_abs >= threshold_W2).float()
        
        # Apply masks to the main model
        with torch.no_grad():
            main_model.layer1.weight.data.mul_(mask_W1)
            main_model.layer2.weight.data.mul_(mask_W2)
            main_model.layer1.weight.data *= amp_factor
            main_model.layer2.weight.data *= amp_factor

        print("Applied magnitude-based mask to main model.")

    def save_nfm(self, epoch):
        self.nfm1 = torch.mm(self.layer1.weight.data.t(), self.layer1.weight.data)
        self.nfm2 = torch.mm(self.layer2.weight.data.t(), self.layer2.weight.data)
        print("saving NFMs.")
        plt.figure(figsize=(6, 6))
        sns.heatmap(self.nfm1.detach().cpu().numpy(), cmap="coolwarm")
        plt.title(f'Epoch {epoch} - Layer 1')
        plt.xlabel('y')
        plt.ylabel('x')
        filename = f"results_mlp/Heatmap/Epoch-{epoch}-Layer-1.png"
        plt.savefig(filename)
        self.saved_filenames_1.append(filename)
        plt.close()  # Close the figure to save memory

        plt.figure(figsize=(6, 6))
        sns.heatmap(self.nfm2.detach().cpu().numpy(), cmap="coolwarm")
        plt.title(f'Epoch {epoch} - Layer 2')
        plt.xlabel('y')
        plt.ylabel('x')
        filename = f"results_mlp/Heatmap/Epoch-{epoch}-Layer-2.png"
        plt.savefig(filename)
        self.saved_filenames_2.append(filename)
        plt.close()  # Close the figure to save memory

        

        

class SimpleMLP_LoRA(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, scale=1, rank = 16, switch_epoch = 10, beta = 1.0, init_rank = None):
        super(SimpleMLP_LoRA, self).__init__()
        self.D = input_dim
        self.N = hidden_dim
        self.scale = scale
        self.rank = rank
        self.layer1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.layer2 = nn.Linear(hidden_dim, output_dim, bias=False)
        # upstream unbalanced initialization
        if beta is not None:
          self.layer1.weight.data *= beta

        # Freeze the original weights initially
        for param in self.layer1.parameters():
            param.requires_grad = False
        for param in self.layer2.parameters():
            param.requires_grad = False

        self.A1, self.B1 = self.spectral_initialize(self.layer1, rank)
        self.A2, self.B2 = self.spectral_initialize(self.layer2, rank)
        # Register the low-rank matrices as trainable parameters
        self.A1 = nn.Parameter(self.A1)
        self.B1 = nn.Parameter(self.B1)
        self.A2 = nn.Parameter(self.A2)
        self.B2 = nn.Parameter(self.B2)

        W1 = torch.matmul(self.A1, self.B1)
        W1.data *= scale
        W2 = torch.matmul(self.A2, self.B2)
        W2.data *= scale

        self.effective_weights1 = W1.t()
        self.effective_weights2 = W2.t()
        self.nfm1 = torch.mm(self.effective_weights1.data.t(), self.effective_weights1.data)
        self.nfm2 = torch.mm(self.effective_weights2.data.t(), self.effective_weights2.data)


    def forward(self, x):
        W1 = torch.matmul(self.A1, self.B1)
        self.effective_weights1 = W1.t()
        x = torch.matmul(x, self.effective_weights1)
        x = x**2

        W2 = torch.matmul(self.A2, self.B2)
        self.effective_weights2 = W2.t()
        x = torch.matmul(x, self.effective_weights2)
        self.nfm1 = torch.mm(self.effective_weights1.data.t(), self.effective_weights1.data)
        self.nfm2 = torch.mm(self.effective_weights2.data.t(), self.effective_weights2.data)
        return x  # Scale the final output


    def update_nfm_and_effective_weights(self):
        self.effective_weights1 = self.layer1.weight.clone().detach()
        self.effective_weights2 = self.layer2.weight.clone().detach()
        self.nfm1 = torch.mm(self.effective_weights1.data.t(), self.effective_weights1.data)
        self.nfm2 = torch.mm(self.effective_weights2.data.t(), self.effective_weights2.data)

    """
    def switch(self):
        # Unfreeze the original weights initially
        # switch to normal full parameters training
        with torch.no_grad():
          W1 = self.layer1.weight.t() + torch.matmul(self.A1, self.B1)
          self.layer1.weight.data = W1.t()
          W2 = self.layer2.weight.t() + torch.matmul(self.A2, self.B2)
          self.layer2.weight.data = W2.t()

        for param in self.layer1.parameters():
            param.requires_grad = True
        for param in self.layer2.parameters():
            param.requires_grad = True

        self.A1.requires_grad = False
        self.B1.requires_grad = False
        self.A2.requires_grad = False
        self.B2.requires_grad = False

    def initialize_low_rank(self, layer, rank):
        # Get the weight matrix of the layer
        weight = layer.weight.data

        # Perform SVD on the attention weights matrix
        U, S, V = torch.svd(weight)
        
        # Compute the sum of all singular values (original nuclear norm)
        original_nuclear_norm = S.sum()
        
        # Retain only the top 'rank' singular values
        S_reduced = S[:rank]
        U = U[:, :rank]
        V = V[:, :rank]
        
        # Compute the sum of the reduced singular values (current nuclear norm)
        reduced_nuclear_norm = S_reduced.sum()
        
        # Scale the reduced singular values to preserve the original nuclear norm
        scaling_factor = original_nuclear_norm / reduced_nuclear_norm
        S_reduced = S_reduced * scaling_factor
        
        # Recompose the matrix with the scaled singular values
        S_reduced = torch.diag(S_reduced)
        low_rank_weight = torch.mm(U, torch.mm(S_reduced, V.t()))
        
        # Set the layer's weight to the low-rank approximation
        layer.weight.data = low_rank_weight
    """

    def spectral_initialize(self, layer, rank):
        # Get the weight matrix of the layer
        weight = layer.weight.data
        # Perform SVD on the attention weights matrix
        U, S, V = torch.svd(weight)
        # Select the top 'rank' components
        U = U[:, :rank]
        V = V[:, :rank]
        S = S[:rank]

        # Compute square roots of singular values
        sqrt_S = torch.diag(S.sqrt())

        # Compute U * sqrt(S) and sqrt(S) * V^T
        U_sqrt_S = U @ sqrt_S
        sqrt_S_V = sqrt_S @ V.T

        return U_sqrt_S, sqrt_S_V




def generate_data(p, task = 'mul'):
    X = []
    y = []
    for x in range(p):
        for y_val in range(p):
            v_x = np.zeros(p)
            v_y = np.zeros(p)
            v_x[x] = 1
            v_y[y_val] = 1
            
            v_xy = np.concatenate([v_x, v_y])
            if task == 'mul':
                z = (x * y_val) % p
            elif task == 'add':
                z = (x + y_val) % p
            elif task == 'hard':
                z = (x**2 + x * y_val + y_val**2) % p
            
            v_z = np.zeros(p)
            v_z[z] = 1
       
            X.append(v_xy)
            y.append(z)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y


# Function to compute the Jacobian
def compute_jacobian(model, device, x, wrt='parameters'):
    """
    Compute the Jacobian of the model output with respect to the input or model parameters.
    
    Parameters:
    model (nn.Module): The neural network model.
    x (torch.Tensor): The input data with requires_grad=True.
    wrt (str): Compute Jacobian with respect to 'input' or 'parameters'.
    
    Returns:
    torch.Tensor: The computed Jacobian.
    """
    # Forward pass to compute the output
    model = model.to(device)
    x = x.to(device)
    output = model(x)

    # Check if the output contains NaN or inf
    if torch.isnan(output).any():
        raise ValueError("Model output contains NaN values.")
    if torch.isinf(output).any():
        raise ValueError("Model output contains inf values.")
    
    jacobian = []
    param_list = [p for p in model.parameters() if p.requires_grad]
    
    # Loop over each element in the output
    for i in range(output.shape[1]):
        # Compute the gradient of the i-th element of the output
        grad_output = torch.zeros_like(output)
        grad_output[:, i] = 1
        grad_output = grad_output.to(device)
        
        if wrt == 'inputs':
            jacobian_row = torch.autograd.grad(outputs=output, inputs=x, grad_outputs=grad_output, retain_graph=True, create_graph=True)[0]
        elif wrt == 'parameters':
            jacobian_row = torch.autograd.grad(outputs = output, inputs= param_list , grad_outputs=grad_output, retain_graph=True)
            jacobian_row = torch.cat([g.flatten() for g in jacobian_row])
        else:
            raise ValueError("wrt must be 'inputs' or 'parameters'")

        # Check if the gradient contains NaN or inf
        if torch.isnan(jacobian_row).any():
            raise ValueError(f"Jacobian row {i} contains NaN values.")
        if torch.isinf(jacobian_row).any():
            raise ValueError(f"Jacobian row {i} contains inf values.")
        
        jacobian.append(jacobian_row)
    
    # Stack the rows to form the Jacobian matrix
    jacobian = torch.stack(jacobian)
    # flatten all output dimensions to 1d vector
    jacobian = torch.cat([g.flatten() for g in jacobian])
    return jacobian

# Function to compute NTK for a batch of data
def compute_ntk_batch(device, jacobian):
    """
    Compute the NTK matrix for a given batch of data.
    
    Parameters:
    model (nn.Module): The neural network model.
    device (torch.device): The device to run computations on.
    data_batch (torch.Tensor): The input data batch.
    
    Returns:
    torch.Tensor: The computed NTK matrix for the batch.
    """
    n = jacobian.size(0)
    ntk_batch = torch.zeros((n, n), device=device)
    # Compute NTK entries
    for i in range(n):
        for j in range(n):
            ntk_batch[i, j] = torch.dot(jacobian[i].view(-1), jacobian[j].view(-1))
    
    return ntk_batch

def generate_orthogonal_vectors(p, seed=42):
    np.random.seed(seed)  # Set the random seed for reproducibility
    # Generate a random vector for mu1
    mu1 = np.random.randn(p)
    mu1 = mu1 / np.linalg.norm(mu1)  # Normalize to unit length

    # Generate a random vector for mu2 orthogonal to mu1
    mu2 = np.random.randn(p)
    mu2 = mu2 - mu2.dot(mu1) * mu1  # Remove component in direction of mu1
    mu2 = mu2 / np.linalg.norm(mu2)  # Normalize to unit length

    mu1 *= 10.0
    mu2 *= 10.0

    return mu1, mu2

def generate_XOR_data(n_samples, p, eta=0.05, seed = 42):
    """
    Parameters:
    - n_samples: Number of samples to generate.
    - p: Dimension of the feature space (R^p).
    - mu1: Mean vector for the first cluster (shape: [p]).
    - mu2: Mean vector for the second cluster (shape: [p]).
    - eta: Label flipping probability (0 <= eta < 0.5).

    Returns:
    - X: Generated data points (shape: [n_samples, p]).
    - y: Labels (shape: [n_samples]).
    """
    mu1, mu2 = generate_orthogonal_vectors(p)
    assert mu1.shape[0] == p and mu2.shape[0] == p, "mu1 and mu2 must have dimension p"
    assert np.isclose(np.dot(mu1, mu2), 0), "mu1 and mu2 must be orthogonal"

    np.random.seed(seed)
    
    # Step 1: Generate clean labels y_tilde ~ Unif{-1, 1}
    y = np.random.choice([-1, 1], size=n_samples)
    # Step 2: Generate input data X based on the clean labels
    X_noisy = np.zeros((n_samples, p))
    for i in range(n_samples):
        if y[i] == 1:
            X_noisy[i] = 0.5 * np.random.multivariate_normal(+mu1, np.eye(p)) + \
                   0.5 * np.random.multivariate_normal(-mu1, np.eye(p))
        else:
            X_noisy[i] = 0.5 * np.random.multivariate_normal(+mu2, np.eye(p)) + \
                   0.5 * np.random.multivariate_normal(-mu2, np.eye(p))
    
    # Step 3: Introduce label noise (flip labels with probability eta)
    noise = np.random.binomial(1, eta, size=n_samples)  # 1 means flip the label
    y_noisy = np.where(noise == 1, -y, y)  # Flip labels based on noise

    X_noisy = np.array(X_noisy, dtype=np.float32)
    y_noisy = np.array(y_noisy, dtype=np.int64)

    # Step 4: generate the true clean distribution

    X_clean = np.zeros((n_samples, p))
    for i in range(n_samples):
        if y[i] == 1:
            X_clean[i] = 0.5 * np.random.multivariate_normal(+mu1, np.eye(p)) + \
                   0.5 * np.random.multivariate_normal(-mu1, np.eye(p))
        else:
            X_clean[i] = 0.5 * np.random.multivariate_normal(+mu2, np.eye(p)) + \
                   0.5 * np.random.multivariate_normal(-mu2, np.eye(p))
            
    X_clean = np.array(X_clean, dtype=np.float32)
    y_clean = np.array(y, dtype=np.int64)
    
    return X_clean, X_noisy, y_clean, y_noisy
