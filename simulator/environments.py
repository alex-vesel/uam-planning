import numpy as np
import cv2
import pickle
import copy
from profilehooks import profile

from .maps import RandomMap
from .agent import eVTOL, eVTOLObservation
from .plotter import Plotter
from .utils import *

class Environment:
    def __init__(self,
                 map,
                 config,
                 init=True,
    ):
        self.map = map
        self.config = config
        self.time = 0
        self.dt = config.D_T
        self.event_cooldown = np.zeros((self.config.N_AGENTS, self.config.N_AGENTS))
        self.has_new_passenger = [False for _ in range(self.config.N_AGENTS)]
        self.delivered_passenger = [False for _ in range(self.config.N_AGENTS)]

        self.LOS_matrix = np.zeros((self.config.N_AGENTS, self.config.N_AGENTS))
        self.NMAC_matrix = np.zeros((self.config.N_AGENTS, self.config.N_AGENTS))

        # recording attributes
        self.LOS_events = 0
        self.NMAC_events = 0
        self.trip_distances = []
        self.passengers_served = 0
        self.passenger_wait_times = []

        if init:
            self.init_agents()
            self.plotter = Plotter(self)
            self.get_distances()


    def init_agents(self):
        self.agents = []
        for i in range(self.config.N_AGENTS):
            x, y = np.random.rand(2) * self.map.size
            self.agents.append(eVTOL(i, self.config.MAX_SPEED_KMS, self.config.MAX_ACCEL_KMS, x, y))

        
    def reset(self):
        self.time = 0
        self.init_agents()
        self.map.reset()
        self.get_distances()
        return self._get_observations()


    def get_distances(self):
        # check agent collisions
        self.agent_agent_distances = pairwise_distance(self.agents, self.agents)
        self.agent_agent_heading = pairwise_heading(self.agents, self.agents)
        
        # check agent vertiport proximity
        self.agent_vertiport_distances = pairwise_distance(self.agents, self.map.vertiports)
        self.agent_vertiport_heading = pairwise_heading(self.agents, self.map.vertiports)
        self.num_passengers = [len(vertiport.passengers) for vertiport in self.map.vertiports]


    def check_collisions(self):
        # check agent collisions
        # for i, agent in enumerate(self.agents):
        #     for j, other_agent in enumerate(self.agents):
        #         if j <= i:
        #             continue
        #         self.event_cooldown[i, j] = max(0, self.event_cooldown[i, j] - self.dt)
        #         if self.event_cooldown[i, j] == 0:
        #             if self.agent_agent_distances[i, j] < self.config.NMAC_DIST:
        #                 self.NMAC_events += 1
        #                 self.event_cooldown[i, j] = self.config.EVENT_COOLDOWN
        #             if self.agent_agent_distances[i, j] < self.config.LOS_DIST:
        #                 self.LOS_events += 1
        #                 self.event_cooldown[i, j] = self.config.EVENT_COOLDOWN

        # check agent collisions       
        self.LOS_matrix = self.agent_agent_distances
        np.fill_diagonal(self.LOS_matrix, np.inf)
        self.LOS_matrix = self.LOS_matrix < self.config.LOS_DIST

        self.NMAC_matrix = self.agent_agent_distances
        np.fill_diagonal(self.NMAC_matrix, np.inf)
        self.NMAC_matrix = self.NMAC_matrix < self.config.NMAC_DIST

        self.event_cooldown = np.maximum(0, self.event_cooldown - self.dt)
        grounded_agents = np.array([agent.grounded for agent in self.agents])
        # give grounded agents a cooldown
        self.event_cooldown[grounded_agents] = 20
        self.event_cooldown[:, grounded_agents] = 20

        self.LOS_matrix = self.LOS_matrix & (self.event_cooldown == 0)
        self.NMAC_matrix = self.NMAC_matrix & (self.event_cooldown == 0)
        LOS_idx = np.argwhere(self.LOS_matrix)
        NMAC_idx = np.argwhere(self.NMAC_matrix)
        self.LOS_events += len(LOS_idx)
        self.NMAC_events += len(NMAC_idx)

        self.event_cooldown[LOS_idx[:, 0], LOS_idx[:, 1]] = self.config.EVENT_COOLDOWN


    def step(self, actions, step_map=True, verbose=False):
        # if verbose:
        #     print(actions)

        # step map
        if step_map:
            self.map.step()

        if verbose:
            print(self.map.agent_passenger_matching)

        # step each agent
        for i, (agent, action) in enumerate(zip(self.agents, actions)):
            passenger = agent.passenger

            if agent.passenger and self.agent_vertiport_distances[i][agent.passenger.destination] < self.config.VERTIPORT_RADIUS_KM:
                nearest_vp = agent.passenger.destination
            # check if target is close
            elif self.agent_vertiport_distances[i][agent.target] < self.config.VERTIPORT_RADIUS_KM:
                nearest_vp = agent.target
            else:
                nearest_vp = np.argmin(self.agent_vertiport_distances[i])

            # Agent landing
            if action.land and self.agent_vertiport_distances[i][nearest_vp] < self.config.VERTIPORT_RADIUS_KM:
                # if verbose:
                #     print(f'Agent {i} landed at vertiport {nearest_vp}')
                #     print(self.map.vertiports[nearest_vp].passengers)
                delivered_passenger = self.map.vertiports[nearest_vp].land(agent)
                if delivered_passenger:
                    passenger = None
                    self.passengers_served += 1
                    self.trip_distances.append((agent.trip_distance, self.map.vp_distances[agent.passenger.origin, agent.passenger.destination]))

            if agent.id in self.map.agent_passenger_matching:
                passenger = self.map.agent_passenger_matching[agent.id]
                self.passenger_wait_times.append(self.time - passenger.start_time)

            if not action.is_flight_action and action.stay and not passenger:
                self.map.vertiports[nearest_vp].ground(agent)
            agent.step(self.dt, action, passenger=passenger)

        self.get_distances()
        self.check_collisions()

        self.time += self.dt

        return self._get_observations()
    

    def simple_step(self, dt):
        # perform a step ahead in time given passenger assigments simplifying environment logic
        for i, agent in enumerate(self.agents):
            # distance travelled
            dist_travelled = agent.speed * dt
            target_dist = self.agent_vertiport_distances[i][agent.target]
            if dist_travelled > target_dist:
                agent.x = self.map.vertiports[agent.target].x
                agent.y = self.map.vertiports[agent.target].y
                delivered_passenger = self.map.vertiports[agent.target].land(agent)
                if delivered_passenger:
                    self.passengers_served += 1
                    agent.passenger = None
            else:
                # advance agent greedily towards target
                agent.x += dist_travelled * np.cos(self.agent_vertiport_heading[i][agent.target])
                agent.y += dist_travelled * np.sin(self.agent_vertiport_heading[i][agent.target])

        self.map.step()

        # assign agents new passengers
        for i, agent in enumerate(self.agents):
            if agent.id in self.map.agent_passenger_matching:
                print('lookahead passenger')
                agent.passenger = self.map.agent_passenger_matching[agent.id]

        self.time += self.dt


    def update_dt(self, dt):
        self.dt = dt
        self.map.update_dt(dt)
    

    def finish(self):
        self.trip_distances = np.array(self.trip_distances)
        if len(self.trip_distances) > 0:
            self.trip_ratio = np.mean(self.trip_distances[:, 0] / self.trip_distances[:, 1])
        else:
            self.trip_ratio = 0
        self.NMAC_events_h = self.NMAC_events / (self.time / 3600)
        self.LOS_events_h = self.LOS_events / (self.time / 3600)
        self.passengers_h = self.passengers_served / (self.time / 3600)
        # add remaining passengers to wait times
        for vp in self.map.vertiports:
            for passenger in vp.passengers:
                self.passenger_wait_times.append(self.time - passenger.start_time)
        self.avg_wait_time = np.mean(self.passenger_wait_times)
        self.max_wait_time = np.max(self.passenger_wait_times)


    def done(self):
        return self.time > self.config.MAX_TIME or (self.map.done() and self.passengers_served >= self.config.MAX_PASSENGERS)


    def _get_observations(self):
        observations = []
        LOS_events = np.any(self.LOS_matrix, axis=1)
        for i, agent in enumerate(self.agents):
            observations.append(eVTOLObservation(agent, self.agent_vertiport_distances[i], self.agent_vertiport_heading[i], self.num_passengers, self.config.VERTIPORT_RADIUS_KM, self.agent_agent_distances[i], self.agent_agent_heading[i], LOS_events[i]))
        return observations

    
    def plot(self):
        self.plotter.plot()


    def __deepcopy__(self, memo):
        new_env = Environment(copy.deepcopy(self.map), self.config, init=False)
        new_env.agents = pickle.loads(pickle.dumps(self.agents, -1))
        new_env.time = self.time
        new_env.agent_vertiport_distances = pickle.loads(pickle.dumps(self.agent_vertiport_distances, -1))
        new_env.agent_vertiport_heading = pickle.loads(pickle.dumps(self.agent_vertiport_heading, -1))

        return new_env




if __name__ == '__main__':
    max_speed = ms_to_kms(90)   # m/s
    max_accel = ms_to_kms(10)   # m/s^2
    n_agents = 10                # number of agents
    map_size = 100               # km
    n_vertiports = 10           # number of vertiports
    d_t = 10                     # s


    map = RandomMap(map_size, n_vertiports)
    agents = []
    for i in range(n_agents):
        x, y = np.random.rand(2) * map_size
        agents.append(eVTOL(i, max_speed, max_accel, x, y))
    simulator = Environment(map, d_t=d_t, agents=agents)
    for _ in range(100):
        simulator.step()
        simulator.plot()