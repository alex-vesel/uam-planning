import numpy as np
import copy
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from murty import Murty

class HungarianMatching():
    def __init__(self, observations, env, commit=True):
        self.obs = observations
        self.env = env
        self.commit = commit

        self.prepare()


    def get_vertiport_scores(self):
        self.vertiport_scores = np.zeros(len(self.env.map.vertiports))
        for i, vp in enumerate(self.env.map.vertiports):
            self.vertiport_scores[i] = vp.arrival_rate_s
        self.vertiport_scores /= np.sum(self.vertiport_scores) + 1e-6

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


    def prepare(self):
        self.passengers = copy.deepcopy([vp.get_passengers() for vp in self.env.map.vertiports])

        # remove passengers that are already assigned
        if self.commit:
            assigned_passengers = []
            for i, agent in enumerate(self.env.agents):
                if agent.passenger_target is not None:
                    assigned_passengers.append(agent.passenger_target)
            for i, passengers in enumerate(self.passengers):
                for passenger in assigned_passengers:
                    if passenger in passengers:
                        passengers.remove(passenger)
              
        self.get_vertiport_scores()

        self.vp_targets_template = []
        self.passenger_targets_template = []
        self.unassigned_agents_template = []
        self.flight_levels = [0 for _ in self.obs]
        for i, obs in enumerate(self.obs):
            if obs.passenger:
                self.vp_targets_template.append(obs.passenger.destination)
                self.passenger_targets_template.append(None)
                self.unassigned_agents_template.append(obs.id)
            else:
                if self.commit and self.env.agents[i].passenger_target is not None:
                    self.vp_targets_template.append(self.env.agents[i].passenger_target.origin)
                    self.passenger_targets_template.append(self.env.agents[i].passenger_target)
                else:
                    self.vp_targets_template.append(None)
                    self.passenger_targets_template.append(None)
                    self.unassigned_agents_template.append(obs.id)

        self.cum_jobs = []
        self.vp_lookup = []
        for i, passengers in enumerate(self.passengers):
            if passengers:
                self.cum_jobs.append(len(passengers))
                self.vp_lookup.append(i)

        self.cum_jobs = np.cumsum(self.cum_jobs)
        if len(self.cum_jobs) != 0:
            self.num_jobs = self.cum_jobs[-1]

            self.get_costs()
            # convert costs to int
            self.costs = (self.costs * 10).astype(int)
            if self.costs.shape[0] > 0 and self.costs.shape[1] > 0:
                if self.costs.shape[1] < self.costs.shape[0]:
                    unique_cols = np.unique(self.costs, axis=1)
                    self.m = Murty(self.costs.T)
                else:
                    self.m = Murty(self.costs)
                self.ok = True
            else:
                self.ok = False

        self.prev_cost = 1e10
        self.get_agents_per_vp()


    def match(self):
        self.vp_targets = copy.copy(self.vp_targets_template)
        self.passenger_targets = copy.copy(self.passenger_targets_template)
        self.unassigned_agents = copy.copy(self.unassigned_agents_template)
        # print(self.unassigned_agents)

        # murtys algo
        unassigned_indices = []
        matches = []
        if len(self.cum_jobs) != 0:
            if self.costs.shape[1] < self.costs.shape[0]:
                its = 0
                cost = self.prev_cost
                while abs(cost - self.prev_cost) < 0.00001 and its < 100 and self.ok:
                    self.ok, cost, sol = self.m.draw()
                    its += 1
                # print(cost, self.prev_cost, abs(cost - self.prev_cost))
                self.prev_cost = cost
                if not self.ok:
                    self.assign_unassigned_agents()
                    return None, None
                # print(self.env.time, cost, sol)
                unassigned_indices = sol.tolist()
                matches = list(range(self.costs.shape[0]))
            else:
                its = 0
                cost = self.prev_cost
                while abs(cost - self.prev_cost) < 0.00001 and its < 100 and self.ok:
                    self.ok, cost, sol = self.m.draw()
                    its += 1
                # print(cost, self.prev_cost, abs(cost - self.prev_cost))
                self.prev_cost = cost
                if not self.ok:
                    self.assign_unassigned_agents()
                    return None, None
                matches = sol.tolist()
                unassigned_indices = list(range(self.costs.shape[0]))

            for unassigned_idx, match in zip(unassigned_indices, matches):
                # get the vertiport index from cumulative jobs
                jobs_idx = np.argmax(self.cum_jobs > match)
                vertiport_idx = self.vp_lookup[jobs_idx]
                passenger_idx = match - self.cum_jobs[jobs_idx - 1] if jobs_idx > 0 else match
                if self.vp_targets[self.unassigned_agents[unassigned_idx]] is None:
                    self.vp_targets[self.unassigned_agents[unassigned_idx]] = vertiport_idx
                if self.passengers[vertiport_idx]:
                    if self.passengers[vertiport_idx][passenger_idx]:
                        self.passenger_targets[self.unassigned_agents[unassigned_idx]] = self.passengers[vertiport_idx][passenger_idx]

        # for agents still unassigned, assign to nearest vertiport
        self.assign_unassigned_agents()
                # self.vp_targets[i] = np.argmin(self.env.agent_vertiport_distances[i])

        return self.vp_targets, self.passenger_targets
    

    def assign_unassigned_agents(self):
        for i, target in enumerate(self.vp_targets):
            if target is None:
                if self.commit:
                    self.vp_targets[i] = np.argmin(self.env.agent_vertiport_distances[i])
                else:
                    self.vp_targets[i] = int(self.agent_to_vp[i])
    

    def get_agents_per_vp(self):
        # get vertiport number of agents relative to score
        self.num_agents_per_vp = len(self.obs) * self.vertiport_scores
        self.num_agents_per_vp = np.round(self.num_agents_per_vp).astype(int)
        if np.sum(self.num_agents_per_vp) < len(self.obs):
            self.num_agents_per_vp[np.argmax(self.num_agents_per_vp)] += len(self.obs) - np.sum(self.num_agents_per_vp)
        self.nominal_dist = copy.copy(self.num_agents_per_vp)
        # get current distribution of agents 
        self.delta_dist = copy.copy(self.num_agents_per_vp)
        for i in range(len(self.obs)):
            nearest_vp = np.argmin(self.env.agent_vertiport_distances[i])
            self.delta_dist[nearest_vp] -= 1
        # get nearest number of agents to vertiport so distribution is even
        self.agent_to_vp = np.ones(len(self.obs)) * -1
        assigned_agents = []
        agents_assigned = 0
        distances_mat = copy.copy(self.env.agent_vertiport_distances)
        while agents_assigned < len(self.obs):
            for i, count in enumerate(self.num_agents_per_vp):
                if count > 0:
                    # get closest agent not assigned
                    distances = distances_mat[:, i]
                    distances[assigned_agents] = np.inf
                    agent = np.argmin(distances)
                    self.agent_to_vp[agent] = i
                    assigned_agents.append(agent)

                    agents_assigned += 1
                    self.num_agents_per_vp[i] -= 1



    def get_costs(self):
        costs = np.ones((len(self.unassigned_agents_template), self.num_jobs)) * 100000
        for row_idx, agent_id in enumerate(self.unassigned_agents_template):
            cur_target = self.vp_targets_template[agent_id]
            col_idx = 0
            for i, passengers in enumerate(self.passengers):
                if cur_target is not None:
                    dist = self.env.agent_vertiport_distances[agent_id][cur_target] + \
                        self.env.map.vp_distances[cur_target, i]
                    # if another free agent is closer 
                    # if np.argmin(self.env.agent_vertiport_distances[:, cur_target]) != i:
                    #     dist += 10000
                else:
                    dist = self.env.agent_vertiport_distances[agent_id][i]
                if passengers:
                    for j, passenger in enumerate(passengers):
                        costs[row_idx, col_idx] = (dist) / 0.09
                        col_idx += 1
                        
        self.costs = costs