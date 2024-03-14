import numpy as np
import copy
from .hungarian_matching import HungarianMatching
from policy.greedy_policy import GreedyPolicy

class LookaheadMatching():
    def __init__(self, observations, env):
        self.obs = observations
        self.env = env


    def match(self):
        self.match_obj = HungarianMatching(self.obs, self.env)

        # if self.env.time > 3000:
        #     import IPython; IPython.embed(); exit(0)
        visited_vp_targets = []
        min_los = np.inf
        min_idx = 0
        for j in range(100):
            vp_targets, passenger_targets = self.match_obj.match()
            if vp_targets is None:
                continue
            # elif vp_targets in visited_vp_targets:
            #     continue
            else:
                visited_vp_targets.append(vp_targets)

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
            # print(j, np.mean(np.abs(delta_dist)))

            if np.mean(np.abs(delta_dist)) < min_los:
                min_idx = j
                min_los = np.mean(np.abs(delta_dist))
                self.vp_targets = vp_targets
                self.passenger_targets = passenger_targets

        # print(min_idx)
            
            # new_env = copy.deepcopy(self.env)
            # obs = copy.deepcopy(self.obs)
            # new_env.update_dt(10)
            # all_vp_targets = [vp_targets]

            # new_vp_targets = vp_targets
            # new_passenger_targets = passenger_targets
            # for _ in range(1):
            #     actions = []
            #     for i, agent in enumerate(new_env.agents):
            #         if agent.passenger:
            #             agent.target = agent.passenger.destination
            #             obs[i].update_target(agent.passenger.destination)
            #         else:
            #             agent.target = new_vp_targets[i]
            #             agent.passenger_target = new_passenger_targets[i]
            #             obs[i].update_target(new_vp_targets[i])
            #         actions.append(GreedyPolicy(obs[i], i, new_env).search())
            #     obs = new_env.step(actions, step_map=False)

            #     new_match_obj = HungarianMatching(obs, new_env)
            #     new_vp_targets, new_passenger_targets = new_match_obj.match()
            #     all_vp_targets.append(new_vp_targets)

            # if new_env.LOS_events - self.env.LOS_events < min_los:
            #     min_idx = j
            #     min_los = new_env.LOS_events - self.env.LOS_events
            #     self.vp_targets = vp_targets
            #     self.passenger_targets = passenger_targets

            # print(j, new_env.LOS_events - self.env.LOS_events)
            # print(visited_vp_targets)

            # print(self.env.time, np.array(all_vp_targets))

            # new_match_obj = HungarianMatching(obs, new_env)
            # new_vp_targets, new_passenger_targets = new_match_obj.match()

        # print(min_idx)
        # for i, agent in enumerate(self.obs):
        #     if agent.passenger:
        #         new_vp_targets[i] = agent.passenger.destination
        #     elif new_env.agents[i].passenger:
        #         new_vp_targets[i] = new_env.agents[i].passenger.origin
        
        # self.vp_targets = new_vp_targets
        # self.passenger_targets = new_passenger_targets