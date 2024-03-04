import numpy as np
import copy
import pickle
from sklearn.cluster import KMeans

from .vertiports import Vertiport
from .utils import *


class RandomMap:
    def __init__(self, size, n_vertiports, dt, max_passengers, init=True):
        self.size = size
        self.n_vertiports = n_vertiports
        self.dt = dt
        self.max_passengers = max_passengers
        self.vertiports = []
        self.agent_passenger_matching = {}
        self.max_distance = np.sqrt(2 * size**2)
        if init:
            self.generate_map()


    def generate_map(self):
        locations = np.random.rand(self.n_vertiports, 2) * self.size
        # correlate arrival rates with distance
        kmeans = KMeans(n_clusters=self.n_vertiports // 3 + 1, random_state=0).fit(locations)
        # get random arrival rate for each cluster
        cluster_arrival_rates = np.random.normal(10, 5, self.n_vertiports // 3 + 1)
        cluster_arrival_rates = np.clip(cluster_arrival_rates, 2, 24).astype(int)
        arrival_rates = np.zeros(self.n_vertiports)
        # assign arrival rates to each vertiport
        for i in range(self.n_vertiports):
            vp_arrival_rate = np.random.normal(cluster_arrival_rates[kmeans.labels_[i]], 2)
            arrival_rates[i] = np.clip(vp_arrival_rate, 1, 28).astype(int)

        for i in range(self.n_vertiports):
            x, y = locations[i]
            self.vertiports.append(Vertiport(i, x, y, self.dt, self.n_vertiports, arrival_rates[i]))

        self.vp_distances = pairwise_distance(self.vertiports, self.vertiports)


    def reset(self):
        self.vertiports = []
        self.generate_map()


    def update_dt(self, dt):
        self.dt = dt
        for vertiport in self.vertiports:
            vertiport.dt = dt


    def step(self):
        self.agent_passenger_matching = {}
        # if self.done_generating():
            # print('done generating')
            # print([vp.passengers for vp in self.vertiports])
            # print([vp.total_passengers for vp in self.vertiports])
        # if self.done():
            # print('done')
        for vertiport in self.vertiports:
            matches = vertiport.step(done_generating=self.done_generating())
            for id, passenger in matches:
                self.agent_passenger_matching[id] = passenger


    def done(self):
        return all([len(vp.passengers) == 0 for vp in self.vertiports]) and \
            np.sum([vp.total_passengers for vp in self.vertiports]) >= self.max_passengers
    

    def done_generating(self):
        return np.sum([vp.total_passengers for vp in self.vertiports]) >= self.max_passengers


    def __deepcopy__(self, memo):
        new_map = RandomMap(self.size, self.n_vertiports, self.dt, self.max_passengers, init=False)
        new_map.vertiports = pickle.loads(pickle.dumps(self.vertiports, -1))
        new_map.vp_distances = self.vp_distances
        
        return new_map


if __name__ == '__main__':
    map = RandomMap(50, 10)
    print(map.vertiports)