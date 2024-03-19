import numpy as np
import copy
import pickle
from sklearn.cluster import KMeans
import matplotlib.image as mpimg
from scipy.ndimage import gaussian_filter

from .vertiports import Vertiport
from .utils import *


class Map:
    def __init__(self, config, init=True): 
        self.config = config
        self.size = config.MAP_SIZE
        self.n_vertiports = config.N_VERTIPORTS
        self.dt = config.D_T
        self.max_passengers = config.MAX_PASSENGERS
        self.vertiports = []
        self.agent_passenger_matching = {}
        self.max_distance = np.sqrt(2 * config.MAP_SIZE ** 2)
        self.arrival_rate = config.ARRIVAL_RATE
        if init:
            self.generate_map()


    def reset(self):
        pass
        # self.vertiports = []
        # self.generate_map()


    def update_dt(self, dt):
        self.dt = dt
        for vertiport in self.vertiports:
            vertiport.dt = dt


    def step(self, step=True):
        self.agent_passenger_matching = {}
        for vertiport in self.vertiports:
            matches = vertiport.step(done_generating=self.done_generating(), step_passengers=step)
            for id, passenger in matches:
                self.agent_passenger_matching[id] = passenger


    def generate_passengers(self):
        # pregenerate passenger arrival times for consistent testing
        while not self.done_generating():
            for vertiport in self.vertiports:
                vertiport.generate_passengers()
        # reset time
        for vertiport in self.vertiports:
            vertiport.time = 0
            vertiport.total_passengers = 0


    def done(self):
        return all([len(vp.cur_passengers) == 0 for vp in self.vertiports]) and \
            np.sum([vp.total_passengers for vp in self.vertiports]) >= self.max_passengers
    

    def done_generating(self):
        return np.sum([vp.total_passengers for vp in self.vertiports]) >= self.max_passengers
    

class RandomMap(Map):
    def __init__(self, config, init=True):
        super().__init__(config, init=init)


    def generate_map(self):
        locations = np.random.rand(self.n_vertiports, 2) * self.size
        # correlate arrival rates with distance
        kmeans = KMeans(n_clusters=self.n_vertiports // 3 + 1, random_state=0).fit(locations)
        # get random arrival rate for each cluster
        cluster_arrival_rates = np.random.normal(10, 5, self.n_vertiports // 3 + 1)
        cluster_arrival_rates = np.clip(cluster_arrival_rates, 2, 200).astype(int)
        arrival_rates = np.zeros(self.n_vertiports)
        # assign arrival rates to each vertiport
        for i in range(self.n_vertiports):
            vp_arrival_rate = np.random.normal(cluster_arrival_rates[kmeans.labels_[i]], 2)
            arrival_rates[i] = np.clip(vp_arrival_rate, 1, 200).astype(int)

        for i in range(self.n_vertiports):
            x, y = locations[i]
            self.vertiports.append(Vertiport(i, x, y, self.dt, self.n_vertiports, arrival_rates[i]))

        self.generate_passengers()

        self.vp_distances = pairwise_distance(self.vertiports, self.vertiports)


    def __deepcopy__(self, memo):
        new_map = RandomMap(self.config, init=False)
        new_map.vertiports = pickle.loads(pickle.dumps(self.vertiports, -1))
        new_map.vp_distances = self.vp_distances
        
        return new_map
    

class SatMap(Map):
    def __init__(self, config, init=True):
        if init:
            if config.MAP_TYPE == 'sf':
                sat_pop = np.load('./simulator/sf_sat_pop.npy')
            elif config.MAP_TYPE == 'nyc':
                sat_pop = np.load('./simulator/nyc_sat_pop.npy')
            sat_pop = sat_pop[:min(sat_pop.shape[:2]), :min(sat_pop.shape[:2])]
            self.sat = sat_pop[:, :, :3]
            self.pop = sat_pop[:, :, 3]
            self.pop_density = sat_pop[:, :, 4]
            # smooth population density to remove noise
            self.pop_density = gaussian_filter(self.pop_density, 30)
            self.max_x, self.max_y = self.pop.shape
        super().__init__(config, init=init)


    def generate_map(self):
        self.generate_vertiports()
        self.generate_passengers()
        self.vp_distances = pairwise_distance(self.vertiports, self.vertiports)

    
    def generate_vertiports(self):
        coords = np.zeros((2, self.n_vertiports))
        # sample based on pop_density_prob, then zero out region according to required population per vertiport
        pop_per_vp = np.sum(self.pop) / self.n_vertiports
        pop_prob = self.get_pop_prob()
        for i in range(self.n_vertiports):
            sample_locs = np.random.choice(np.arange(self.pop.size), size=1, p=pop_prob)
            x, y = np.unravel_index(sample_locs, self.pop.shape)
            x, y = x[0], y[0]
            coords[0, i], coords[1, i] = x, y
            # zero out region near coords until pop_density, grow largest radius
            radius = 50
            circle = self.get_circle(x, y, radius)
            while np.sum(self.pop[circle[:, 0], circle[:, 1]]) < pop_per_vp and radius < 150:
                radius += 15
                circle = self.get_circle(x, y, radius)
            self.pop[circle[:, 0], circle[:, 1]] = 0

            pop_prob = self.get_pop_prob()

        # get density of population around each vertiport
        vp_densities = np.zeros(self.n_vertiports)
        for i in range(self.n_vertiports):
            x, y = coords[:, i]
            vp_densities[i] = self.pop_density[int(x), int(y)]

        # assign arrival rates to each vertiport
        arrival_rates = np.clip(self.arrival_rate * vp_densities / np.sum(vp_densities), 1, 200).astype(int)
        print("Vertiport arrival rates: ", arrival_rates)

        self.vertiports = []
        for i in range(self.n_vertiports):
            x_orig, y_orig = coords[:, i]
            y = x_orig / self.max_x * self.size
            x = y_orig / self.max_y * self.size
            self.vertiports.append(Vertiport(i, x, y, self.dt, self.n_vertiports, arrival_rates[i]))


    def get_circle(self, x, y, radius):
        circle = []
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                if i**2 + j**2 <= radius**2 and 0 <= x + i < self.max_x and 0 <= y + j < self.max_y:
                    circle.append((x + i, y + j))
        return np.array(circle)
    

    def get_pop_prob(self):
        pop_prob = self.pop / np.sum(self.pop)
        pop_prob = pop_prob/ np.sum(pop_prob)
        return pop_prob.flatten()


    def __deepcopy__(self, memo):
        new_map = SatMap(self.config, init=False)
        new_map.vertiports = pickle.loads(pickle.dumps(self.vertiports, -1))
        new_map.vp_distances = self.vp_distances
        new_map.sat = self.sat
        
        return new_map


if __name__ == '__main__':
    map = RandomMap(50, 10)
    print(map.vertiports)