import numpy as np

class MatchingScore:
    def __init__(self, observations, env):
        self.obs = observations
        self.env = env

# give points for moving towards passengers that have been waiting for a long time
    def score(self):
        self.env.passengers_served = 0
        self.env.update_dt(100)
        self.env.simple_step(100)
        print(self.env.passengers_served)