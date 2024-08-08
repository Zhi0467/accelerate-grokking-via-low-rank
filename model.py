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

class Block(nn.Module):
    """Causal transformer block
    """

    def __init__(self, dim, num_heads, rank = None):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.ln_2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.rank = rank
        if rank is not None:
          print(f"using rank {rank} for in_proj_matrix.")
          self.attn.in_proj_weight.data = self.low_rank_approximation(self.attn.in_proj_weight, rank)



    def forward(self, x):
        attn_mask = torch.full(
            (len(x), len(x)), -float("Inf"), device=x.device, dtype=x.dtype
        )
        attn_mask = torch.triu(attn_mask, diagonal=1)
        attn_mask[torch.isnan(attn_mask)] = 0.0 # fixes all 'nan' on 'mps' device

        x = self.ln_1(x)
        a, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x
    
    def low_rank_approximation(self, matrix, rank):
        # Perform SVD on the attention weights matrix
        U, S, V = torch.svd(matrix)
        # Retain only the top 'rank' singular values
        S = torch.diag(S[:rank])
        U = U[:, :rank]
        V = V[:, :rank]
        # Recompose the matrix with reduced rank
        low_rank_matrix = torch.mm(U, torch.mm(S, V.t()))
        return low_rank_matrix


class Decoder(nn.Module):
    """Causal Transformer decoder
    """

    def __init__(self, dim=128, num_layers=2, num_heads=4, num_tokens=97, seq_len=5, rank = None):
        super().__init__()
        self.token_embeddings = nn.Embedding(num_tokens, dim)
        self.position_embeddings = nn.Embedding(seq_len, dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(Block(dim, num_heads, rank))

        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_tokens, bias=False)

    def forward(self, x):
        h = self.token_embeddings(x)
        positions = torch.arange(x.shape[0], device=x.device).unsqueeze(-1)
        h = h + self.position_embeddings(positions).expand_as(h)
        for layer in self.layers:
            h = layer(h)

        h = self.ln_f(h)
        logits = self.head(h)
        return logits


def multiplication_mod_p_data(p, eq_token, op_token):
    """x◦y = x/y (mod p) for 0 ≤ x < p, 0 < y < p
    """
    x = torch.arange(p)
    y = torch.arange(1, p)
    x, y = torch.cartesian_prod(x, y).T

    eq = torch.ones_like(x) * eq_token
    op = torch.ones_like(x) * op_token
    result = (x**2 + x * y + y**2 ) % p

    # "All of our experiments used a small transformer trained on datasets of
    # equations of the form a◦b = c, where each of “a”, “◦”, “b”, “=”, and “c”
    # is a seperate token"
    return torch.stack([x, op, y, eq, result])

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, scale=1, rank=None):
        super(SimpleMLP, self).__init__()
        self.D = input_dim
        self.N = hidden_dim
        self.scale = scale
        self.rank = rank

        self.layer1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.layer2 = nn.Linear(hidden_dim, output_dim, bias=False)

        if rank is not None:
          print(f"set init rank to be {rank}.")
          self.initialize_low_rank(self.layer1, rank)
          self.initialize_low_rank(self.layer2, rank)

    def forward(self, x):
        x = self.layer1(x)
        x = x**2 
        x = self.layer2(x)
        return x * self.scale

    def initialize_low_rank(self, layer, rank):
        # Get the weight matrix of the layer
        weight = layer.weight.data

        # Perform SVD on the weight matrix
        U, S, V = torch.svd(weight)

        # Retain only the top 'rank' singular values
        S = torch.diag(S[:rank])
        U = U[:, :rank]
        V = V[:, :rank]

        # Recompose the weight matrix with reduced rank
        low_rank_weight = torch.mm(U, torch.mm(S, V.t()))

        # Set the layer's weight to the low-rank approximation
        layer.weight.data = low_rank_weight

class SimpleMLP_LoRA(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, scale=1, rank = 16, switch_epoch = 10, init_rank = None):
        super(SimpleMLP_LoRA, self).__init__()
        self.D = input_dim
        self.N = hidden_dim
        self.scale = scale
        self.rank = rank
        self.layer1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.layer2 = nn.Linear(hidden_dim, output_dim, bias=False)
        self.epoch = 0
        self.switch_epoch = switch_epoch

        # Freeze the original weights initially
        for param in self.layer1.parameters():
            param.requires_grad = False
        for param in self.layer2.parameters():
            param.requires_grad = False

        # Initialize low-rank matrices for fc1
        self.A1 = nn.Parameter(torch.zeros(input_dim, rank))
        self.B1 = nn.Parameter(torch.randn(rank, hidden_dim))

        # Initialize low-rank matrices for fc2
        self.A2 = nn.Parameter(torch.zeros(hidden_dim, rank))
        self.B2 = nn.Parameter(torch.randn(rank, output_dim))

        if init_rank is not None:
          print(f"set init rank to be {init_rank}.")
          self.initialize_low_rank(self.layer1, init_rank)
          self.initialize_low_rank(self.layer2, init_rank)

        self.effective_weights1 = self.layer1.weight.clone().detach()
        self.effective_weights2 = self.layer2.weight.clone().detach()


    def forward(self, x):
        # First layer with low-rank adaptation
        if self.epoch < self.switch_epoch:
          W1 = self.layer1.weight.t() + torch.matmul(self.A1, self.B1)
          self.effective_weights1 = W1.t()
          x = torch.matmul(x, W1)
          x = x**2

          # Second layer with low-rank adaptation
          W2 = self.layer2.weight.t() + torch.matmul(self.A2, self.B2)
          self.effective_weights2 = W2.t()
          x = torch.matmul(x, W2)
          return x * self.scale  # Scale the final output
        elif self.epoch == self.switch_epoch:
          self.switch()
          x = self.layer1(x)
          x = x**2
          x = self.layer2(x)
          self.effective_weights1 = self.layer1.weight.clone().detach()
          self.effective_weights2 = self.layer2.weight.clone().detach()
          return x * self.scale  # Scale the final output
        else:
          x = self.layer1(x)
          x = x**2
          x = self.layer2(x)
          self.effective_weights1 = self.layer1.weight.clone().detach()
          self.effective_weights2 = self.layer2.weight.clone().detach()
          return x * self.scale  # Scale the final output

    
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

        # Perform SVD on the weight matrix
        U, S, V = torch.svd(weight)

        # Retain only the top 'rank' singular values
        S = torch.diag(S[:rank])
        U = U[:, :rank]
        V = V[:, :rank]

        # Recompose the weight matrix with reduced rank
        low_rank_weight = torch.mm(U, torch.mm(S, V.t()))

        # Set the layer's weight to the low-rank approximation
        layer.weight.data = low_rank_weight



# Data generation function
def generate_data(p):
    X = []
    y = []
    for x in range(p):
        for y_val in range(p):
            v_x = np.zeros(p + 1)
            v_y = np.zeros(p + 1)
            v_x[x] = 1
            v_y[y_val] = 1
            v_y[-1] = 1
            
            v_xy = np.concatenate([v_x, v_y])
            z = (x * y_val) % p
            
            v_z = np.zeros(p)
            v_z[z] = 1
       
            X.append(v_xy)
            y.append(z)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y

def generate_data_without_positional_labels(p):
    X = []
    y = []
    for x in range(p):
        for y_val in range(p):
            v_x = np.zeros(p)
            v_y = np.zeros(p)
            v_x[x] = 1
            v_y[y_val] = 1
            
            v_xy = np.concatenate([v_x, v_y])
            z = (x * y_val) % p
            
            v_z = np.zeros(p)
            v_z[z] = 1
       
            X.append(v_xy)
            y.append(z)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y

def generate_data_of_the_hard_task(p):
    X = []
    y = []
    for x in range(p):
        for y_val in range(p):
            v_x = np.zeros(p)
            v_y = np.zeros(p)
            v_x[x] = 1
            v_y[y_val] = 1
            
            v_xy = np.concatenate([v_x, v_y])
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
        
        jacobian.append(jacobian_row)
    
    # Stack the rows to form the Jacobian matrix
    jacobian = torch.stack(jacobian)
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