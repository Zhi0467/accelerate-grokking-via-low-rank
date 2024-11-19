import subprocess

configs = [ 
{
        "label": "3",
        "filter": "none",
        "p": 97,
        "weight_decay": 0.1,
        "optimizer": "AdamW",
        "alpha": 0.98,
        "lamb": 2.0,
        "batch_size": 128,
        "lr": 1e-3,
        "hidden_dim": 256,
        "fraction": 0.5,
        "init_scale": 16,
        "num_epochs": 2000,
        "init_rank": 1,
        "low_rank_switch": False,
        "save_weights": False,
},
{
        "label": "4",
        "filter": "ema",
        "p": 97,
        "weight_decay": 0.1,
        "optimizer": "AdamW",
        "alpha": 0.98,
        "lamb": 2.0,
        "batch_size": 128,
        "lr": 1e-3,
        "hidden_dim": 256,
        "fraction": 0.5,
        "init_scale": 16,
        "num_epochs": 2000,
        "init_rank": 1,
        "low_rank_switch": False,
        "save_weights": False,
},
]

# Function to run the main program with specified arguments
def run_experiment_in_metainit(config):
    cmd = ["python", "metainit.py"]
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
for config in configs:
    run_experiment_in_metainit(config)

