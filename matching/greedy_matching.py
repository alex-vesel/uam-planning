import numpy as np

class GreedyVertiportMatching:
    # assigns each evtol to the nearest vertiport
    def __init__(self, observations, env=None):
        self.obs = observations
        self.passenger_targets = [None for _ in observations]

    def match(self):
        self.vp_targets = []
        for obs in self.obs:
            if obs.passenger:
                self.vp_targets.append(obs.passenger.destination)
            else:
                self.vp_targets.append(np.argmin(obs.vp_distances))


class GreedyPassengerMatching:
    # assigns each evtol to the nearest vertiport with passengers
    def __init__(self, observations, env):
        self.obs = observations
        self.passenger_targets = [None for _ in observations]

    def match(self):
        self.vp_targets = []
        for obs in self.obs:
            if obs.passenger:
                self.vp_targets.append(obs.passenger.destination)
            else:
                # get closest vertiport with passengers
                min_dist = np.inf
                target = np.argmin(obs.vp_distances)
                for i, d in enumerate(obs.vp_distances):
                    if d < min_dist and obs.vp_num_passengers[i] > 0:
                        min_dist = d
                        target = i
                self.vp_targets.append(target)