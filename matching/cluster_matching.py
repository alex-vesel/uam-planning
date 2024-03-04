import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

class ClusterMatching():
    def __init__(self, observations, env):
        self.obs = observations
        self.env = env

        self.get_vertiport_scores()


    def get_vertiport_scores(self):
        # get future expected vertiport score given arrival rate and distance to other vertiports
        self.vertiport_scores = []
        for vertiport in self.env.map.vertiports:
            score = 0
            distances = []
            arrival_rates = []
            for i, vp in enumerate(self.env.map.vertiports):
                if vp != vertiport:
                    score += (1 - self.env.map.vp_distances[vertiport.id, vp.id] / self.env.map.max_distance) * vp.arrival_rate
            final_score = vertiport.arrival_rate + 0.2 * score
            self.vertiport_scores.append(final_score)
        # print(self.vertiport_scores / np.max(self.vertiport_scores))
        # print softmax of scores
        self.vertiport_scores = np.exp(self.vertiport_scores) / np.sum(np.exp(self.vertiport_scores)).squeeze()


    def match(self):
        self.waiting_times = [vp.get_waiting_times() for vp in self.env.map.vertiports]
        self.get_vertiport_scores()
        
        self.targets = []
        self.unassigned_agents = []
        for obs in self.obs:
            if obs.passenger:
                self.targets.append(obs.passenger.destination)
            else:
                self.targets.append(None)
                # self.unassigned_agents.append(obs.id)
            self.unassigned_agents.append(obs.id)

        cum_jobs = []
        for waiting_times in self.waiting_times:
            # num_jobs += len(waiting_times) if waiting_times else 1
            if waiting_times:
                cum_jobs.append(len(waiting_times))
            else:
                cum_jobs.append(1)
        cum_jobs = np.cumsum(cum_jobs)
        num_jobs = cum_jobs[-1]

        costs = np.zeros((len(self.unassigned_agents), num_jobs))
        for row_idx, agent_id in enumerate(self.unassigned_agents):
            cur_target = self.targets[agent_id]
            col_idx = 0
            for i, waiting_times in enumerate(self.waiting_times):
                if cur_target is not None:
                    dist = self.env.agent_vertiport_distances[agent_id][cur_target] + \
                        self.env.map.vp_distances[cur_target, i]
                else:
                    dist = self.env.agent_vertiport_distances[agent_id][i]
                if waiting_times:
                    for j, waiting_time in enumerate(waiting_times):
                        costs[row_idx, col_idx] = dist / 0.09# - waiting_time
                        col_idx += 1
                else:
                    costs[row_idx, col_idx] = dist / 0.09 + 10000
                    # costs[row_idx, col_idx] -= self.vertiport_scores[i] * 100
                    col_idx += 1

        # match agents to vertiports
        matches = linear_sum_assignment(costs)[1]

        for i, match in enumerate(matches):
            # get the vertiport index from cumulative jobs
            vertiport_idx = np.argmax(cum_jobs > match)
            if self.targets[self.unassigned_agents[i]] is None:
                self.targets[self.unassigned_agents[i]] = vertiport_idx

        # for agents still unassigned, assign to nearest vertiport
        for i, target in enumerate(self.targets):
            if target is None:
                self.targets[i] = np.argmin(self.env.agent_vertiport_distances[i])

