import subprocess

configs1 = [ 
{
        "label": "1",
        "filter": "none",
        "p": 97,
        "weight_decay": 0.1,
        "optimizer": "AdamW",
        "alpha": 0.95,
        "lamb": 5.0,
        "batch_size": 256,
        "lr": 5e-3,
        "hidden_dim": 256,
        "fraction": 0.5,
        "init_scale": 32.0,
        "num_epochs": 1000,
        "sparse_init": 'random',
        "sparsity": 0.8,
        "low_rank_switch": False,
        "save_weights": False,
},
{
        "label": "2",
        "filter": "none",
        "p": 97,
        "weight_decay": 0.1,
        "optimizer": "AdamW",
        "alpha": 0.95,
        "lamb": 5.0,
        "batch_size": 256,
        "lr": 5e-3,
        "hidden_dim": 256,
        "fraction": 0.5,
        "switch_epoch": 20,
        "init_scale": 32.0,
        "num_epochs": 1000,
        "low_rank_switch": True,
        "switch_to_rank": 2,
        "save_weights": False,
},
]

# Function to run the main program with specified arguments
def run_experiment_in_main_mlp(config):
    cmd = ["python", "main_mlp.py"]
    for key, value in config.items():
        if isinstance(value, bool):
            if value:  # Only add the flag if it's set to True
                cmd.append(f"--{key}")
        else:
            cmd.append(f"--{key}")
            cmd.append(str(value))
    print("Running command:", " ".join(cmd))
    subprocess.run(cmd)


# Run experiments with different configurations
for config in configs1:
    run_experiment_in_main_mlp(config)

