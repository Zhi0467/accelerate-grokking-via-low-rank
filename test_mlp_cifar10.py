import subprocess

configs1 = [ 
{
        "label": "1",
        "filter": "none",
        "weight_decay": 0.001,
        "optimizer": "AdamW",
        "alpha": 0.95,
        "lamb": 2.0,
        "batch_size": 64,
        "lr": 4e-3,
        "hidden_dim": 256,
        "init_scale": 1.0,
        "num_epochs": 2000,
        "enable_lr_update": True,
        "update_rank_percentage": 0.1,
        "save_weights": False,
},
{
        "label": "2",
        "filter": "none",
        "weight_decay": 0.001,
        "optimizer": "AdamW",
        "alpha": 0.95,
        "lamb": 2.0,
        "batch_size": 128,
        "lr": 4e-3,
        "hidden_dim": 64,
        "init_scale": 1.0,
        "init_rank": 4,
        "num_epochs": 2000,
        "enable_lr_update": True,
        "update_rank_percentage": 0.1,
        "save_weights": False,
},
]

# Function to run the main program with specified arguments
def run_experiment_in_main_mlp_cifar10(config):
    cmd = ["python", "main_mlp_cifar10.py"]
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
    run_experiment_in_main_mlp_cifar10(config)

