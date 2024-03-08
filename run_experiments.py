import os
import copy
import json
import numpy as np
import pandas as pd
from simulate import run_experiment
from config import Config
from utils import get_exp_dir


# experiment configuration string
exp_dir = get_exp_dir(Config)
out_dir = os.path.join('output', exp_dir)
os.makedirs(out_dir, exist_ok=True)

# run experiments
for i in range(10):
    np.random.seed(i)
    exp_config = copy.deepcopy(Config)
    exp_config.PLOT = False

    # run experiment
    env = run_experiment(exp_config)

    # save results
    exp_json = {
        "time": float(env.time),
        "passengers_served": float(env.passengers_served),
        "NMAC_events": float(env.NMAC_events),
        "LOS_events": float(env.LOS_events),
        "NMAC_events_h": float(env.NMAC_events_h),
        "LOS_events_h": float(env.LOS_events_h),
        "trip_ratio": float(env.trip_ratio),
        "passengers_h": float(env.passengers_h),
        "avg_wait_time": float(env.avg_wait_time),
        "max_wait_time": float(env.max_wait_time)
    }

    with open(os.path.join(out_dir, f"exp_{i}.json"), "w") as f:
        json.dump(exp_json, f)

# open results and get avg/std
results = []
for i in range(3):
    with open(os.path.join(out_dir, f"exp_{i}.json"), "r") as f:
        results.append(json.load(f))

results = pd.DataFrame(results)

for col in results.columns:
    print(f"{col}: {results[col].mean()} +/- {results[col].std()}")