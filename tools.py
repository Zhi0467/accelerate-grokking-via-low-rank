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
