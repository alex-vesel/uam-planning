import os
import copy
import json
import pickle
import numpy as np
import pandas as pd

exp_paths = os.walk('./output')

for root, dirs, files in exp_paths:
    # check if first of dirs is an integer
    try:
        int(dirs[0])
    except:
        continue

    if "nyc" not in root:
        continue
    if "flight_levels_1" not in root:
        continue

    print(root, len(dirs))
    num_agents = int(root.split("_")[7].split("/")[0])

    exp_times = []
    exp_passengers_served = []
    exp_nmacs = []
    exp_losses = []
    exp_nmacs_h = []
    exp_losses_h = []
    exp_passengers_h = []
    exp_trip_distances = []
    exp_avg_wait_times = []
    exp_max_wait_times = []

    for dir in dirs:
        exp_dir = os.path.join(root, dir)

        # open json
        with open(os.path.join(exp_dir, f"exp.json"), "r") as f:
            exp_json = json.load(f)

        # open pickles
        trip_distances = pickle.load(open(os.path.join(exp_dir, f"trip_distances.pkl"), "rb"))

        # append to lists
        exp_times.append(exp_json["time"])
        exp_passengers_served.append(exp_json["passengers_served"])
        exp_nmacs.append(exp_json["NMAC_events"] / 2)
        exp_losses.append(exp_json["LOS_events"] / 2)
        exp_nmacs_h.append(exp_json["NMAC_events_h"] / 2)
        exp_losses_h.append(exp_json["LOS_events_h"] / 2)
        exp_passengers_h.append(exp_json["passengers_h"])
        exp_trip_distances.append(trip_distances)
        exp_avg_wait_times.append(exp_json["avg_wait_time"])
        exp_max_wait_times.append(exp_json["max_wait_time"])

    trip_distances = np.concatenate(exp_trip_distances)
    trip_ratios = trip_distances[:, 0] / trip_distances[:, 1]


    # calculate avg/std
    print("Time: ", np.round(np.mean(exp_times), 1), "pmpm", np.round(np.std(exp_times), 1))
    # print("Passengers served: ", np.round(np.mean(exp_passengers_served), np.round(np.std(exp_passengers_served))
    # print("NMACs: ", np.round(np.mean(exp_nmacs), np.round(np.std(exp_nmacs))
    # print("LOSs: ", np.round(np.mean(exp_losses), np.round(np.std(exp_losses))
    print("NMACs / (h agent): ", np.round(np.mean(exp_nmacs) / num_agents, 3), "pmpm", np.round(np.std(exp_nmacs) / num_agents, 3))
    print("LOSs / (h agent): ", np.round(np.mean(exp_losses) / num_agents, 3), "pmpm", np.round(np.std(exp_losses) / num_agents, 3))
    print("Passengers / (h agent): ", np.round(np.mean(exp_passengers_h) / num_agents, 1), "pmpm", np.round(np.std(exp_passengers_h) / num_agents, 1))
    print("Avg wait time: ", np.round(np.mean(exp_avg_wait_times), 1), "pmpm", np.round(np.std(exp_avg_wait_times), 1))
    print("Max wait time: ", np.round(np.mean(exp_max_wait_times), 1), "pmpm", np.round(np.std(exp_max_wait_times), 1))
    print("Trip ratio: ", np.round(np.mean(trip_ratios), 3), "pmpm", np.round(np.std(trip_ratios), 3))
    print()



