import subprocess

# This script allows multiple runs at once.
main_configs = [
{
        "label": "2",
        "filter": "none",
        "optimizer": "AdamW",
        "budget": 200000,
        "p": 113,
        "weight_decay": 0.1,
        "alpha": 0.95,
        "lamb": 2.0,
        "batch_size": 2048,
        "lr": 4e-3,
        "lambda_nuc": 0,
        "lambda_rank": 0,
        "save_weights": False,
        "enable_lr_update": False,
        "init_rank": 4,
},
]


# Define different argument configurations
configs = [
     {
        "label": "test",
        "filter": "none",
        "budget": 300000,
        "p": 97,
        "batch_size": 2048,
        "two_stage": False,
    },
    {
        "label": "test",
        "filter": "ema",
        "weight_decay": 0.005,
        "alpha": 0.98,
        "lamb": 2.0,
        "budget": 300000,
        "batch_size": 2048,
        "p": 97,
        "two_stage": False,
    },
    {
        "label": "test",
        "filter": "ma",
        "weight_decay": 0.01,
        "window_size": 100,
        "lamb": 5.0,
        "budget": 300000,
        "batch_size": 2048,
        "p": 97,
        "two_stage": False,
    },
    {
        "label": "test",
        "filter": "ma",
        "weight_decay": 0.01,
        "window_size": 100,
        "lamb": 5.0,
        "budget": 300000,
        "batch_size": 2048,
        "p": 97,
        "two_stage": True,
    },
    # Add more configurations as needed
]

# run config2 to compare 1.AdamW with weight decay 1
# 2. grokfast ema with Adam and a small weight decay
# 3. grokfast ema with AdamW with weight decay 1 
# 4. AdamW with weight decay 0.005
# 5. Adam with weight decay 1
# everything else being the same.
configs2 = [
    {
        "label": "test",
        "filter": "none",
        "optimizer": "AdamW",
        "budget": 300000,
        "p": 113,
        "weight_decay": 1.0,
        "save_weights": True,
    },
    {
        "label": "test",
        "filter": "ema",
        "weight_decay": 0.005,
        "alpha": 0.98,
        "lamb": 2.0,
        "budget": 300000,
        "p": 113,
        "save_weights": True,
    },
    {
        "label": "test",
        "filter": "ema",
        "weight_decay": 1,
        "optimizer": "AdamW",
        "alpha": 0.98,
        "lamb": 2.0,
        "budget": 300000,
        "p": 113,
        "save_weights": True,
    },
    {
        "label": "test",
        "filter": "none",
        "optimizer": "AdamW",
        "budget": 300000,
        "p": 113,
        "weight_decay": 0.005,
        "save_weights": True,
    },
     {
        "label": "test",
        "filter": "none",
        "optimizer": "Adam",
        "budget": 300000,
        "p": 113,
        "weight_decay": 1,
        "save_weights": True,
    },
]
# run configs3 to test different starting point of grokfast.
# we can only use 
configs3 = [
{
        "label": "test1",
        "filter": "ema",
        "weight_decay": 1.0,
        "optimizer": "AdamW",
        "alpha": 0.98,
        "lamb": 2.0,
        "two_stage": True,
        "save_weights": True,
        "starting_point": 500,
},
{
        "label": "test2",
        "filter": "ema",
        "weight_decay": 1.0,
        "optimizer": "AdamW",
        "alpha": 0.98,
        "lamb": 2.0,
        "two_stage": True,
        "save_weights": True,
        "starting_point": 1000,
},
{
        "label": "test3",
        "filter": "ema",
        "weight_decay": 1.0,
        "optimizer": "AdamW",
        "alpha": 0.98,
        "lamb": 2.0,
        "two_stage": True,
        "save_weights": True,
        "starting_point": 2000,
},
]

# run this config - test smoother with grid search
configs4 = [
  {
        "label": "test1",
        "filter": "smoother",
        "optimizer": "AdamW",
        "weight_decay": 1.0,
        "budget": 200000,
        "batch_size": 1024,
        "lr": 2e-3,
        "p": 97,
        "beta": 0.98,
        "pp": 0.01,
        "two_stage": False,
  },
  {
        "label": "test2",
        "filter": "smoother",
        "optimizer": "AdamW",
        "weight_decay": 1.0,
        "budget": 200000,
        "batch_size": 1024,
        "lr": 2e-3,
        "p": 97,
        "beta": 0.8,
        "pp": 0.05,
        "two_stage": False,
  },
  {
        "label": "test3",
        "filter": "smoother",
        "optimizer": "AdamW",
        "weight_decay": 1.0,
        "budget": 200000,
        "batch_size": 1024,
        "lr": 2e-3,
        "p": 97,
        "beta": 0.7,
        "pp": 0.1,
        "two_stage": False,
  },
  {
        "label": "test4",
        "filter": "smoother",
        "optimizer": "Adam",
        "weight_decay": 0.005,
        "budget": 200000,
        "batch_size": 1024,
        "lr": 2e-3,
        "p": 97,
        "beta": 0.98,
        "pp": 0.01,
        "two_stage": False,
  },
]

# a complimentary test to configs2 on the effects of AdamW
configs5 = [
    {
        "label": "NO1",
        "filter": "none",
        "optimizer": "AdamW",
        "budget": 150000,
        "p": 97,
        "weight_decay": 1.0,
        "batch_size": 2048,
        "lr": 4e-3,
        "save_weights": True,
    },
    {
        "label": "NO2",
        "filter": "ema",
        "weight_decay": 1,
        "alpha": 0.98,
        "lamb": 2.0,
        "optimizer": "AdamW",
        "budget": 150000,
        "p": 97,
        "batch_size": 2048,
        "lr": 4e-3,
        "save_weights": False,
    },
    {
        "label": "NO2",
        "filter": "ema",
        "weight_decay": 1,
        "alpha": 0.98,
        "lamb": 2.0,
        "optimizer": "AdamW",
        "budget": 150000,
        "p": 97,
        "batch_size": 2048,
        "lr": 4e-3,
        "save_weights": True,
    },
    {
        "label": "NO3",
        "filter": "none",
        "weight_decay": 0.01,
        "optimizer": "AdamW",
        "budget": 150000,
        "p": 97,
        "batch_size": 2048,
        "lr": 4e-3,
        "save_weights": True,
    },
    {
        "label": "NO4",
        "filter": "ema",
        "optimizer": "Adam",
        "budget": 150000,
        "p": 97,
        "weight_decay": 0.01,
        "batch_size": 2048,
        "lr": 4e-3,
        "save_weights": True,
    },
]

# run configs6 to test kalman filter
configs6 = [
    {
        "label": "NO1",
        "filter": "kalman",
        "optimizer": "AdamW",
        "budget": 150000,
        "p": 97,
        "weight_decay": 1.0,
        "batch_size": 2048,
        "lr": 4e-3,
        "process_noise": 1e-5,
        "measurement_noise": 1e-3,
        "save_weights": False,
    },
    {
        "label": "NO2",
        "filter": "kalman",
        "optimizer": "AdamW",
        "budget": 150000,
        "p": 97,
        "weight_decay": 1.0,
        "batch_size": 2048,
        "lr": 4e-3,
        "process_noise": 1e-3,
        "measurement_noise": 1e-1,
        "save_weights": False,
    },
    {
        "label": "NO3",
        "filter": "kalman",
        "optimizer": "AdamW",
        "budget": 150000,
        "p": 97,
        "weight_decay": 1.0,
        "batch_size": 2048,
        "lr": 4e-3,
        "process_noise": 1e-2,
        "measurement_noise": 1.0,
        "save_weights": False,
    },
    {
        "label": "NO4",
        "filter": "kalman",
        "optimizer": "Adam",
        "budget": 150000,
        "p": 97,
        "weight_decay": 0.005,
        "batch_size": 2048,
        "lr": 4e-3,
        "process_noise": 1e-3,
        "measurement_noise": 1e-1,
        "save_weights": False,
    },
]
# run this config to test
# 1. grokfast with AdamW and a small wd 0.005
# 2. grokfast with AdamW and a small wd 0.1
# 3. grokfast with Adam and no wd
# task: x^2 + xy + y^2 
# but we stick to the same grokfast parameters
configs7 = [
{
        "label": "NO1",
        "filter": "ema",
        "optimizer": "AdamW",
        "budget": 200000,
        "p": 113,
        "weight_decay": 0.005,
        "alpha": 0.98,
        "lamb": 2.0,
        "save_weights": False,
},
{
        "label": "NO2",
        "filter": "ema",
        "optimizer": "AdamW",
        "budget": 200000,
        "p": 113,
        "weight_decay": 0.1,
        "alpha": 0.98,
        "lamb": 2.0,
        "save_weights": False,
},
{
        "label": "NO3",
        "filter": "ema",
        "optimizer": "Adam",
        "budget": 200000,
        "p": 113,
        "weight_decay": 0,
        "alpha": 0.98,
        "lamb": 2.0,
        "save_weights": False,
},
]

configs8 = [
{
        "label": "NO1",
        "filter": "ema",
        "optimizer": "AdamW",
        "budget": 300000,
        "p": 113,
        "weight_decay": 0.005,
        "alpha": 0.98,
        "lamb": 2.0,
        "batch_size": 2048,
        "lr": 4e-3,
        "save_weights": False,
},
{
        "label": "NO2",
        "filter": "ema",
        "optimizer": "AdamW",
        "budget": 300000,
        "p": 113,
        "weight_decay": 0.1,
        "alpha": 0.98,
        "lamb": 2.0,
        "batch_size": 2048,
        "lr": 4e-3,
        "save_weights": False,
},
{
        "label": "NO3",
        "filter": "ema",
        "optimizer": "AdamW",
        "budget": 300000,
        "p": 113,
        "weight_decay": 1.0,
        "alpha": 0.98,
        "lamb": 2.0,
        "batch_size": 2048,
        "lr": 4e-3,
        "save_weights": False,
},
]

configs9 = [
{
        "label": "NO4",
        "filter": "ema",
        "optimizer": "AdamW",
        "budget": 300000,
        "p": 113,
        "weight_decay": 0,
        "alpha": 0.98,
        "lamb": 2.0,
        "batch_size": 2048,
        "lr": 4e-3,
        "save_weights": False,
},
{
        "label": "NO5",
        "filter": "ema",
        "optimizer": "AdamW",
        "budget": 300000,
        "p": 113,
        "weight_decay": 0.1,
        "alpha": 0.98,
        "lamb": 3.0,
        "batch_size": 2048,
        "lr": 4e-3,
        "save_weights": False,
},
{
        "label": "NO6",
        "filter": "ema",
        "optimizer": "AdamW",
        "budget": 300000,
        "p": 113,
        "weight_decay": 1.0,
        "alpha": 0.98,
        "lamb": 5.0,
        "batch_size": 2048,
        "lr": 4e-3,
        "save_weights": False,
},
]



# In this config: we grid search post_grokfast
# on alpha
# we keep wd = 0.1 so that AdamW doesn't dominate the training
configs11 = [
{
        "label": "tune",
        "filter": "ema",
        "optimizer": "AdamW",
        "budget": 300000,
        "p": 113,
        "weight_decay": 0.1,
        "alpha": 0.85,
        "lamb": 5.0,
        "batch_size": 2048,
        "lr": 4e-3,
        "save_weights": False,
},
{
        "label": "tune",
        "filter": "ema",
        "optimizer": "AdamW",
        "budget": 300000,
        "p": 113,
        "weight_decay": 0.1,
        "alpha": 0.95,
        "lamb": 5.0,
        "batch_size": 2048,
        "lr": 4e-3,
        "save_weights": False,
},
{
        "label": "tune",
        "filter": "ema",
        "optimizer": "AdamW",
        "budget": 300000,
        "p": 113,
        "weight_decay": 0.1,
        "alpha": 0.97,
        "lamb": 5.0,
        "batch_size": 2048,
        "lr": 4e-3,
        "save_weights": False,
},
]

# this one test post-grokfast on simple multiplication
# same params as configs8 but different task.
configs12 = [
{
        "label": "NO1",
        "filter": "ema",
        "optimizer": "AdamW",
        "budget": 300000,
        "p": 97,
        "weight_decay": 0.005,
        "alpha": 0.98,
        "lamb": 2.0,
        "batch_size": 2048,
        "lr": 4e-3,
        "save_weights": False,
},
{
        "label": "NO2",
        "filter": "ema",
        "optimizer": "AdamW",
        "budget": 300000,
        "p": 97,
        "weight_decay": 0.1,
        "alpha": 0.98,
        "lamb": 2.0,
        "batch_size": 2048,
        "lr": 4e-3,
        "save_weights": False,
},
{
        "label": "NO3",
        "filter": "ema",
        "optimizer": "AdamW",
        "budget": 300000,
        "p": 97,
        "weight_decay": 1.0,
        "alpha": 0.98,
        "lamb": 2.0,
        "batch_size": 2048,
        "lr": 4e-3,
        "save_weights": False,
},
]

configs13 = [
    {
        "label": "tune",
        "filter": "ema",
        "optimizer": "AdamW",
        "budget": 300000,
        "p": 113,
        "weight_decay": 0.1,
        "alpha": 0.95,
        "lamb": 10.0,
        "batch_size": 2048,
        "lr": 4e-3,
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
def run_experiment_in_main_transformer(config):
    cmd = ["python", "main_transformer.py"]
    for key, value in config.items():
        if isinstance(value, bool):
            if value:  # Only add the flag if it's set to True
                cmd.append(f"--{key}")
        else:
            cmd.append(f"--{key}")
            cmd.append(str(value))
    print("Running command:", " ".join(cmd))
    subprocess.run(cmd)

def run_experiment_in_main_old(config):
    cmd = ["python", "main_old.py"]
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
for config in main_configs:
    run_experiment_in_main_transformer(config)

"""
for config in sanity_test_config:
    run_experiment_in_main_old(config)
"""

