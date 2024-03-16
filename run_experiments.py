import os
import copy
import json
import pickle
import numpy as np
import pandas as pd
from simulate import run_experiment
from config import Config, config_calculations
from utils import get_exp_dir

policies = ['greedy']
matchings = ['lookahead']
vertiport_agents = [(10, 5), (20, 8), (40, 12), (80, 16), (120, 24), (200, 30), (400, 30)]
arrival_rates = [1]
map_types = ['sf', 'nyc']

def exp_iterator():
    for policy in policies:
        for matching in matchings:
            for N_AGENTS, N_VERTIPORTS in vertiport_agents:
                for ARRIVAL_RATE_SCALE in arrival_rates:
                    for MAP_TYPE in map_types:
                        config = copy.deepcopy(Config)
                        config.POLICY = policy
                        config.MATCHING = matching
                        config.N_AGENTS = N_AGENTS
                        config.N_VERTIPORTS = N_VERTIPORTS
                        if MAP_TYPE == 'nyc':
                            config.N_VERTIPORTS = min(config.N_VERTIPORTS, 16)
                        config.ARRIVAL_RATE_SCALE = ARRIVAL_RATE_SCALE
                        config.MAP_TYPE = MAP_TYPE
                        config.MAX_PASSENGERS = 10 * N_AGENTS
                        config.PLOT = False
                        config_calculations(config)
                        yield config


# run experiments
for config in exp_iterator():
    exp_dir = get_exp_dir(config)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Running experiment with config: {config.__dict__}")
    if config.N_AGENTS > 100:
        num_exp = 5
    else:
        num_exp = 10
    for i in range(num_exp):
        np.random.seed(i)

        # run experiment
        env = run_experiment(config)

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

        run_dir = os.path.join(exp_dir, f"{i}")
        os.makedirs(run_dir, exist_ok=True)

        with open(os.path.join(run_dir, f"exp.json"), "w") as f:
            json.dump(exp_json, f)

        pickle.dump(env.trip_distances, open(os.path.join(run_dir, f"trip_distances.pkl"), "wb"))
        pickle.dump(env.passenger_wait_times, open(os.path.join(run_dir, f"passenger_wait_times.pkl"), "wb"))

    # open results and get avg/std
    results = []
    for i in range(num_exp):
        run_dir = os.path.join(exp_dir, f"{i}")
        with open(os.path.join(run_dir, f"exp.json"), "r") as f:
            results.append(json.load(f))

    results = pd.DataFrame(results)

    for col in results.columns:
        print(f"{col}: {results[col].mean()} +/- {results[col].std()}")