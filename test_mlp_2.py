import subprocess

configs1 = [ 
{
        "label": "3",
        "filter": "none",
        "p": 512,
        "weight_decay": 0.1,
        "optimizer": "AdamW",
        "alpha": 0.98,
        "lamb": 2.0,
        "batch_size": 128,
        "lr": 1e-2,
        "hidden_dim": 512,
        "fraction": 0.5,
        "init_scale": 1e-2,
        "num_epochs": 2000,
        "low_rank_switch": False,
        "save_weights": False,
        "init_rank": 4,
},
{
        "label": "4",
        "filter": "none",
        "p": 512,
        "weight_decay": 0.1,
        "optimizer": "AdamW",
        "alpha": 0.98,
        "lamb": 2.0,
        "batch_size": 128,
        "lr": 1e-2,
        "hidden_dim": 512,
        "fraction": 0.5,
        "init_scale": 1e-2,
        "num_epochs": 2000,
        "low_rank_switch": False,
        "save_weights": False,
        "init_rank": 8,
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

