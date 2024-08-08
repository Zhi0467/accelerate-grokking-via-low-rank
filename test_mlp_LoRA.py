import subprocess

# In this config: we grid search post_grokfast
# on lamb
# we keep wd = 0.1 so that AdamW doesn't dominate the training
# we skip lamb = 2.0 , 3.0 because they are tested in configs 8 and 9
configs1 = [ 
{
        "label": "1",
        "filter": "none",
        "p": 97,
        "weight_decay": 0.1,
        "optimizer": "AdamW",
        "alpha": 0.95,
        "lamb": 5.0,
        "batch_size": 512,
        "lr": 4e-3,
        "hidden_dim": 256,
        "fraction": 0.5,
        "LoRA_rank": 16,
        "init_rank": 1,
        "switch_epoch": 2000,
        "init_scale": 8.0,
        "num_epochs": 1000,
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
        "batch_size": 512,
        "lr": 4e-3,
        "hidden_dim": 256,
        "fraction": 0.5,
        "LoRA_rank": 16,
        "init_rank": 2,
        "switch_epoch": 2000,
        "init_scale": 8.0,
        "num_epochs": 1000,
        "save_weights": False,
},
]

# Function to run the main program with specified arguments
def run_experiment_in_main_mlp_LoRA(config):
    cmd = ["python", "main_mlp_LoRA.py"]
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
    run_experiment_in_main_mlp_LoRA(config)

