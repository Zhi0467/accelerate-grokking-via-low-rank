import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from model import *

def generate_sparse_input(input_dim):
    """Generate a sparse input vector with exactly two random 1s."""
    # Initialize a zero vector of the given input dimension
    input_vector = torch.zeros(input_dim)
    
    # Randomly choose two unique indices to set to 1
    indices = torch.randperm(input_dim)[:2]  # Select 2 unique random indices
    input_vector[indices] = 1.0  # Set these positions to 1
    
    return input_vector

def test_output_scaling(input_dim, hidden_dim, output_dim):
    factor = 0.0
    x = generate_sparse_input(input_dim)
    model = SimpleMLP(input_dim, hidden_dim, output_dim, scale = 16.0)
    low_rank_model = SimpleMLP(input_dim, hidden_dim, output_dim, scale = 64.0, rank = 1)
    with torch.no_grad():
        y = model(x)
        magnitude = y.norm(2).item()  
        ly = low_rank_model(x)
        lmag = ly.norm(2).item()

        factor += magnitude / lmag
    print(f"the output of original model and low-rank model differ by a factor of {factor}")
    return factor


# Parameters
p = 97  # Dimensionality for one-hot vectors
hidden_dim = 256
output_dim = p

# Generate input data
X = generate_sparse_input(p)

# Initialize the model
input_dim = 2 * p  # Because each input is a concatenation of two one-hot vectors
# Run the test
factors = np.array([])
avg = 0.0
for n in range(10000):
    factor = test_output_scaling(input_dim, hidden_dim, output_dim)
    avg += factor
    factors = np.append(factors, factor)

avg /= 10000
print(f"with median {np.median(factors)} and average {avg}")

