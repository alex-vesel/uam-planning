import numpy as np
import copy
import cv2
import matplotlib.pyplot as plt
from .hungarian_matching import HungarianMatching
from policy.greedy_policy import GreedyPolicy

class LookaheadMatching():
    def __init__(self, observations, env):
        self.obs = observations
        self.env = env


    def match(self):
        self.match_obj = HungarianMatching(self.obs, self.env, commit=False)

        flight_levels = []
        grounded_agents = []
        for i, agent in enumerate(self.env.agents):
            if self.obs[i].is_grounded:
                grounded_agents.append(i)
                flight_levels.append(None)
            else:
                flight_levels.append(agent.flight_level)

        vp_targets_list = []
        passenger_targets_list = []
        scores = []
        for j in range(10):
            vp_targets, passenger_targets = self.match_obj.match()
            if vp_targets is None:
                break

            target_dist = [0 for _ in range(len(self.match_obj.num_agents_per_vp))]
            for i, target in enumerate(vp_targets):
                target_dist[target] += 1

            for i, passenger in enumerate(passenger_targets):
                if passenger is not None:
                    target_dist[passenger.destination] += 1
                    target_dist[vp_targets[i]] -= 1

            delta_dist = []
            for i in range(len(self.match_obj.nominal_dist)):
                delta_dist.append(self.match_obj.nominal_dist[i] - target_dist[i])

            vp_targets_list.append(vp_targets)
            passenger_targets_list.append(passenger_targets)
            scores.append(np.mean(np.abs(delta_dist)))

        if len(scores) == 0:
            self.vp_targets = self.match_obj.vp_targets
            self.passenger_targets = self.match_obj.passenger_targets
            for i, level in enumerate(flight_levels):
                if level is None:
                    flight_levels[i] = self.env.agents[i].flight_level
            self.flight_levels = flight_levels
            return
        elif self.env.map.done_generating():
            min_idx = 0
        else:
            min_idx = np.where(scores == np.min(scores))[0][0]

        vp_targets = vp_targets_list[min_idx]
        passenger_targets = passenger_targets_list[min_idx]

        sim_steps = 20
        density_heatmaps = {t: [np.zeros((self.env.map.size, self.env.map.size)) for _ in range(len(grounded_agents))] for t in range(sim_steps)}
        flight_level_heatmaps = {t: [np.zeros((self.env.map.size, self.env.map.size)) for _ in range(self.env.config.N_FLIGHT_LEVELS)] for t in range(sim_steps)}
        agent_positions = {t: [] for t in range(sim_steps)}
        new_env = copy.deepcopy(self.env)
        new_obs = copy.deepcopy(self.obs)
        for t in range(sim_steps):
            actions = []
            for i, agent in enumerate(new_env.agents):
                agent.target = vp_targets[i]
                agent.passenger_target = passenger_targets[i]
                new_obs[i].update_target(vp_targets[i])
                actions.append(GreedyPolicy(new_obs[i], i, new_env).search())
            new_obs = new_env.step(actions, step_map=False)
            for i, obs in enumerate(new_obs):
                obs_x = max(0, min(self.env.map.size - 1, int(obs.x)))
                obs_y = max(0, min(self.env.map.size - 1, int(obs.y)))
                if self.obs[i].is_grounded:
                    density_heatmaps[t][grounded_agents.index(i)][int(obs_x), int(obs_y)] += 1
                    density_heatmaps[t][grounded_agents.index(i)] = cv2.GaussianBlur(density_heatmaps[t][grounded_agents.index(i)], (0, 0), 3 * t / sim_steps + 1)
                else:
                    if not obs.is_grounded:
                        flight_level_heatmaps[t][flight_levels[i]][int(obs_x), int(obs_y)] += 1
                agent_positions[t].append((int(obs_x), int(obs_y)))
            flight_level_heatmaps[t] = [cv2.GaussianBlur(flight_level_heatmaps[t][i], (0, 0), 3 * t / sim_steps + 1) for i in range(self.env.config.N_FLIGHT_LEVELS)]
        
        for i in grounded_agents:
            scores = []
            for flight_level in range(self.env.config.N_FLIGHT_LEVELS):
                score = 0
                for t in range(sim_steps):
                    score += flight_level_heatmaps[t][flight_level][agent_positions[t][i][0], agent_positions[t][i][1]]
                scores.append(score)

            new_flight_level = np.argmin(scores)
            flight_levels[i] = new_flight_level

            if np.min(scores) > sim_steps / 30:
                # delay flight
                vp_targets[grounded_agents.index(i)] = np.argmin(self.env.agent_vertiport_distances[grounded_agents.index(i)])
            else:
                for t in range(sim_steps):
                    flight_level_heatmaps[t][new_flight_level] += density_heatmaps[t][grounded_agents.index(i)]

        self.vp_targets = vp_targets
        self.passenger_targets = passenger_targets
        self.flight_levels = flight_levels
        # self.flight_levels = [0 for _ in self.obs]
        # self.flight_levels = [np.random.randint(0, self.env.config.N_FLIGHT_LEVELS) for _ in self.obs]
        return
        