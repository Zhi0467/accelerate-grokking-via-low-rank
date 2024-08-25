import subprocess

# In this config: we grid search post_grokfast
# on lamb
# we keep wd = 0.1 so that AdamW doesn't dominate the training
# we skip lamb = 2.0 , 3.0 because they are tested in configs 8 and 9
configs10 = [
{
        "label": "1",
        "filter": "none",
        "optimizer": "AdamW",
        "budget": 150000,
        "p": 113,
        "weight_decay": 0.1,
        "alpha": 0.95,
        "lamb": 5.0,
        "batch_size": 2048,
        "lr": 4e-3,
        "large_lr": 4e-3,
        "cutoff_steps": 300,
        "lambda_nuc": 0,
        "lambda_rank": 0,
        "save_weights": False,
},
{
        "label": "2",
        "filter": "ema",
        "optimizer": "AdamW",
        "budget": 150000,
        "p": 113,
        "weight_decay": 0.1,
        "alpha": 0.95,
        "lamb": 5.0,
        "batch_size": 2048,
        "lr": 4e-3,
        "large_lr": 4e-3,
        "cutoff_steps": 300,
        "lambda_nuc": 0,
        "lambda_rank": 0,
        "save_weights": False,
},
]


sanity_test_config = [
  {
        "label": "test",
        "filter": "none",
        "optimizer": "AdamW",
        "budget": 500,
        "p": 113,
        "weight_decay": 1,
        "alpha": 0.98,
        "lamb": 2.0,
        "save_weights": False,
},
]
# Function to run the main program with specified arguments
def run_experiment_in_main_lookahead(config):
    cmd = ["python", "main_lookahead.py"]
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
for config in configs10:
    run_experiment_in_main_lookahead(config)


