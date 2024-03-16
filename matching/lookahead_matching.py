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
        self.match_obj = HungarianMatching(self.obs, self.env)

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
            return
        elif self.env.map.done_generating():
            min_idx = 0
        else:
            min_idx = np.where(scores == np.min(scores))[0][0]

        vp_targets = vp_targets_list[min_idx]
        passenger_targets = passenger_targets_list[min_idx]


        # new_env = copy.deepcopy(self.env)
        # new_obs = copy.deepcopy(self.obs)
        # for _ in range(10):
        #     actions = []
        #     for i, agent in enumerate(new_env.agents):
        #         agent.target = vp_targets[i]
        #         agent.passenger_target = passenger_targets[i]
        #         new_obs[i].update_target(vp_targets[i])
        #         actions.append(GreedyPolicy(new_obs[i], i, new_env).search())
        #     new_obs = new_env.step(actions, step_map=False)
        #     heatmap = np.zeros((new_env.map.size, new_env.map.size))
        #     for i, obs in enumerate(new_obs):
        #         if not obs.is_grounded:
        #             heatmap[int(obs.x), int(obs.y)] += 1
        #     heatmap = cv2.GaussianBlur(heatmap, (0, 0), 3)
        #     plt.imshow(heatmap); plt.show()

        # its = 0
        # while (new_vp_targets != vp_targets or its == 0) and its < 5:
        #     vp_targets = new_vp_targets
        #     passenger_targets = new_passenger_targets

        #     new_env = copy.deepcopy(self.env)
        #     new_obs = copy.deepcopy(self.obs)
        #     for _ in range(1):
        #         actions = []
        #         for i, agent in enumerate(new_env.agents):
        #             agent.target = vp_targets[i]
        #             agent.passenger_target = passenger_targets[i]
        #             new_obs[i].update_target(vp_targets[i])
        #             actions.append(GreedyPolicy(new_obs[i], i, new_env).search())
        #         new_obs = new_env.step(actions, step_map=False)

        #     new_match_obj = HungarianMatching(new_obs, new_env)
        #     new_vp_targets, new_passenger_targets = new_match_obj.match()
        #     for i, agent in enumerate(self.env.agents):
        #         if agent.passenger:
        #             new_vp_targets[i] = agent.passenger.destination
        #         elif new_env.agents[i].passenger:
        #             new_vp_targets[i] = new_env.agents[i].passenger.origin
        #     its += 1
        # # print(its)
        
        self.vp_targets = new_vp_targets
        self.passenger_targets = new_passenger_targets
        return
        
                # new_env.plot()
                # if new_env.LOS_events - self.env.LOS_events > 0:
                #     for obs in new_obs:
                #         if obs.is_conflict and self.obs[obs.id].is_grounded:
                #             vp_targets_list[0][obs.id] = np.argmin(self.obs[obs.id].vp_distances)
            # los_events.append(new_env.LOS_events - self.env.LOS_events)

        # min_idx = np.where(los_events == np.min(los_events))[0][0]
            
        # min_idx = np.where(same_targets)[0]
        # if len(min_idx) == 0:
        #     self.vp_targets = new_vp_targets_0
        #     self.passenger_targets = passenger_targets_list[min_indices[0]]
        #     return
        # else:
        #     min_idx = min_idx[0]
            
        # print(min_idx)
        # min_idx = 0
        # print(min_idx)
        # if np.min(los_events) > 0 and self.env.time > 800:
        #     import IPython; IPython.embed(); exit(0)

        # min_idx = 0

        # self.vp_targets = vp_targets_list[min_indices[min_idx]]
        # self.passenger_targets = passenger_targets_list[min_indices[min_idx]]