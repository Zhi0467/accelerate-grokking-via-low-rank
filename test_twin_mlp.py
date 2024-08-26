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
        "switch_epoch": 50,
        "init_scale": 64.0,
        "num_epochs": 1000,
        "alignment": True,
        "direction_searching_method": 'cbm',
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
        "switch_epoch": 50,
        "init_scale": 64.0,
        "num_epochs": 1000,
        "alignment": True,
        "direction_searching_method": 'mbm',
        "save_weights": False,
},
{
        "label": "3",
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
        "switch_epoch": 50,
        "init_scale": 64.0,
        "alignment": True,
        "switch_to_rank": 1,
        "num_epochs": 1000,
        "direction_searching_method": 'lrds',
        "save_weights": False,
},
{
        "label": "4",
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
        "switch_epoch": 50,
        "init_scale": 64.0,
        "alignment": True,
        "switch_to_rank": 1,
        "num_epochs": 1000,
        "direction_searching_method": 'srds',
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

