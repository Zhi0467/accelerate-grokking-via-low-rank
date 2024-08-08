import subprocess


configs1 = [ 
{
        "label": "1",
        "filter": "ema",
        "p": 97,
        "weight_decay": 0.1,
        "optimizer": "AdamW",
        "alpha": 0.95,
        "lamb": 5.0,
        "batch_size": 512,
        "lr": 4e-3,
        "hidden_dim": 256,
        "fraction": 0.5,
        "init_rank": 2,
        "switch_epoch": 10,
        "init_scale": 8.0,
        "num_epochs": 150,
        "save_weights": False,
},
]

# Function to run the main program with specified arguments
def run_experiment_in_main_twin_mlp(config):
    cmd = ["python", "main_twin_mlp.py"]
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
    run_experiment_in_main_twin_mlp(config)

