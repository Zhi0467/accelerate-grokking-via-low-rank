import subprocess

configs1 = [ 
{
        "label": "1",
        "filter": "none",
        "p": 97,
        "weight_decay": 0,
        "optimizer": "SGD",
        "alpha": 0.95,
        "lamb": 5.0,
        "batch_size": 2048,
        "lr": 10.0,
        "hidden_dim": 256,
        "fraction": 0.5,
        "init_scale": 4.0,
        "num_epochs": 1500,
        "low_rank_switch": False,
        "save_weights": False,
},
{
        "label": "2",
        "filter": "none",
        "p": 97,
        "weight_decay": 0,
        "optimizer": "SGD",
        "alpha": 0.95,
        "lamb": 5.0,
        "batch_size": 2048,
        "lr": 10.0,
        "hidden_dim": 256,
        "fraction": 0.5,
        "init_scale": 4.0,
        "num_epochs": 1500,
        "low_rank_switch": False,
        "save_weights": False,
        "enable_lr_update": True,
        "update_rank_percentage": 0.1,
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

