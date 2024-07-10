import torch
import matplotlib.pyplot as plt
import numpy as np

def load_results(file_path):
    return torch.load(file_path)

def calculate_distances(results1, results2):
    distances_l2 = []
    distances_l1 = []

    for net1, net2 in zip(results1['net'], results2['net']):
        weights1 = torch.cat([p.view(-1) for p in net1.values()])
        weights2 = torch.cat([p.view(-1) for p in net2.values()])
        
        distance_l2 = torch.norm(weights1 - weights2, p=2).item()
        distance_l1 = torch.norm(weights1 - weights2, p=1).item()
        
        distances_l2.append(distance_l2)
        distances_l1.append(distance_l1)

    return distances_l2, distances_l1

def plot_distances(steps, distances_l2, distances_l1, label1, label2):
    plt.figure()
    plt.plot(steps, distances_l2, label="L2 Distance")
    plt.plot(steps, distances_l1, label="L1 Distance")
    plt.xlabel("Optimization Steps")
    plt.ylabel("Distance")
    plt.xscale("log", base=10)
    plt.yscale("log", base=10)
    plt.legend()
    plt.title(f"Distance Between Weights: {label1} vs {label2}")
    plt.savefig(f"results/weights_distance_{label1}_vs_{label2}.png")
    plt.close()

# Load results from both files
label1 = 'damn'
label2 = 'lol'
results1 = load_results(f'results/res_{label1}.pt')
results2 = load_results(f'results/res_{label2}.pt')

# Calculate distances between weights
distances_l2, distances_l1 = calculate_distances(results1, results2)

# Get optimization steps
steps = torch.arange(len(distances_l2)).numpy() * results1['steps_per_epoch']

# Plot distances
plot_distances(steps, distances_l2, distances_l1, label1, label2)
