import numpy as np
import copy
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from .getkBestNoRankHung import getkBestNoRankHung

class ClusterMatching():
    def __init__(self, observations, env):
        self.obs = observations
        self.env = env

        self.get_vertiport_scores()


    def get_vertiport_scores(self):
        self.vertiport_scores = np.zeros(len(self.env.map.vertiports))
        for i, vp in enumerate(self.env.map.vertiports):
            self.vertiport_scores[i] = vp.arrival_rate_s

        vertiport_num_passengers = [len(vp.cur_passengers) for vp in self.env.map.vertiports]

        # get vertiport average
        self.vertiport_avg_scores = []
        for i, vp in enumerate(self.env.map.vertiports):
            # get distance to all other vertiports
            distances = self.env.map.vp_distances[i]
            # weight distances by max possible travel
            distances = distances / self.env.map.max_distance
            distances += 1
            # get average score
            # self.vertiport_avg_scores.append(np.sum(self.vertiport_scores / distances**2))
            self.vertiport_avg_scores.append(np.sum(vertiport_num_passengers / distances))
        self.vertiport_avg_scores -= np.min(self.vertiport_avg_scores)
        self.vertiport_avg_scores /= np.max(self.vertiport_avg_scores) + 1e-6


    def match(self):
        passengers = copy.deepcopy([vp.get_passengers() for vp in self.env.map.vertiports])
        self.waiting_times = [vp.get_waiting_times() for vp in self.env.map.vertiports]

            
        # idx = 0
        # for i, count in enumerate(num_agents_per_vp):
        #     for _ in range(count):
        #         agent_to_vp[idx] = i
        #         idx += 1

        # vp_difference = self.vertiport_scores - self.vertiport_counts
        # print(np.mean(np.abs(vp_difference)))
              
        self.get_vertiport_scores()

        # add "virtual" passengers to vertiports with no passengers
        # for i, passenger in enumerate(passengers):
        #     for _ in range(0):
        #         passenger.append(None)
        #         self.waiting_times[i].append(-vp_difference[i])
                # self.waiting_times[i].append(-self.env.map.vertiports[i].arrival_rate_s)

        self.vp_targets = []
        self.passenger_targets = []
        self.unassigned_agents = []
        for obs in self.obs:
            if obs.passenger:
                self.vp_targets.append(obs.passenger.destination)
            else:
                self.vp_targets.append(None)
                # self.unassigned_agents.append(obs.id)
            self.passenger_targets.append(None)
            self.unassigned_agents.append(obs.id)
        # print(self.unassigned_agents)

        cum_jobs = []
        vp_lookup = []
        for i, waiting_times in enumerate(self.waiting_times):
            # num_jobs += len(waiting_times) if waiting_times else 1
            if waiting_times:
                cum_jobs.append(len(waiting_times))
                vp_lookup.append(i)
            # else:
            #     cum_jobs.append(1)
        cum_jobs = np.cumsum(cum_jobs)
        if len(cum_jobs) != 0:
            num_jobs = cum_jobs[-1]

            # assign penalty for being too far off of even distribution after some time steps
            # only allow delivering agent to pickup passenger if it arrives before you
            # assign costs based on simulation

            # costs = np.zeros((len(self.unassigned_agents), num_jobs))
            costs = np.ones((len(self.unassigned_agents), num_jobs)) * 100000000
            for row_idx, agent_id in enumerate(self.unassigned_agents):
                cur_target = self.vp_targets[agent_id]
                col_idx = 0
                for i, waiting_times in enumerate(self.waiting_times):
                    if cur_target is not None:
                        dist = self.env.agent_vertiport_distances[agent_id][cur_target] + \
                            self.env.map.vp_distances[cur_target, i]
                        # if another free agent is closer 
                        # if np.argmin(self.env.agent_vertiport_distances[:, cur_target]) != i:
                        #     dist += 10000
                    else:
                        dist = self.env.agent_vertiport_distances[agent_id][i]
                    if waiting_times:
                        for j, waiting_time in enumerate(waiting_times):
                            passenger = passengers[i][j]
                            if passenger:
                                passenger_travel_dist = self.env.map.vp_distances[i, passenger.destination]
                            # else:
                            #     passenger_travel_dist = self.env.map.max_distance / 2
                            costs[row_idx, col_idx] = (dist) / 0.09# - waiting_time# * (1 / self.vertiport_scores[passenger.destination])
                            col_idx += 1
                    # else:
                    #     costs[row_idx, col_idx] = 10000000
                    #     # costs[row_idx, col_idx] -= self.vertiport_scores[i] * 100
                    #     col_idx += 1

            # match agents to vertiports

            matches = linear_sum_assignment(costs)
            unassigned_indices = list(matches[0])
            matches = list(matches[1])
            # if self.env.time == 550:
            #     import IPython; IPython.embed(); exit(0)
            # print(linear_sum_assignment(costs))
            # print(costs)

            for unassigned_idx, match in zip(unassigned_indices, matches):
                # get the vertiport index from cumulative jobs
                jobs_idx = np.argmax(cum_jobs > match)
                vertiport_idx = vp_lookup[jobs_idx]
                passenger_idx = match - cum_jobs[jobs_idx - 1] if jobs_idx > 0 else match
                if self.vp_targets[self.unassigned_agents[unassigned_idx]] is None:
                    self.vp_targets[self.unassigned_agents[unassigned_idx]] = vertiport_idx
                if passengers[vertiport_idx]:
                    if passengers[vertiport_idx][passenger_idx]:
                        self.passenger_targets[self.unassigned_agents[unassigned_idx]] = passengers[vertiport_idx][passenger_idx]
        
        
        # for each agent get closet vertiport
        # self.closest_vertiports = []
        # for obs, agent in zip(self.obs, self.env.agents):
        #     if agent.passenger:
        #         self.closest_vertiports.append(agent.passenger.destination)
        #     elif self.passenger_targets[obs.id]:
        #         self.closest_vertiports.append(self.passenger_targets[obs.id].destination)
        #     else:
        #         self.closest_vertiports.append(np.argmin(obs.vp_distances))

        # get count for each vertiport
        # self.vertiport_counts = np.zeros(len(self.env.map.vertiports))
        # for i, target in enumerate(self.closest_vertiports):
        #     self.vertiport_counts[target] += 1

        self.vertiport_scores = self.vertiport_scores / np.sum(self.vertiport_scores)
        # self.vertiport_counts = self.vertiport_counts / np.sum(self.vertiport_counts)

        # get vertiport number of agents relative to score
        num_agents_per_vp = len(self.obs) * self.vertiport_scores
        num_agents_per_vp = np.round(num_agents_per_vp).astype(int)
        if np.sum(num_agents_per_vp) < len(self.obs):
            num_agents_per_vp[np.argmax(num_agents_per_vp)] += len(self.obs) - np.sum(num_agents_per_vp)
        # get nearest number of agents to vertiport so distribution is even
        agent_to_vp = np.ones(len(self.obs)) * -1
        assigned_agents = []
        agents_assigned = 0
        distances_mat = copy.copy(self.env.agent_vertiport_distances)
        while agents_assigned < len(self.obs):
            for i, count in enumerate(num_agents_per_vp):
                if count > 0:
                    # get closest agent not assigned
                    distances = distances_mat[:, i]
                    distances[assigned_agents] = np.inf
                    agent = np.argmin(distances)
                    agent_to_vp[agent] = i
                    assigned_agents.append(agent)

                    agents_assigned += 1
                    num_agents_per_vp[i] -= 1


        # for agents still unassigned, assign to nearest vertiport
        for i, target in enumerate(self.vp_targets):
            if target is None:
                self.vp_targets[i] = int(agent_to_vp[i])
                # self.vp_targets[i] = np.argmin(self.env.agent_vertiport_distances[i])

