import numpy as np
import copy
from .mcts import MCTSMatching

class MCTSHungarianMatching():
    def __init__(self, observations, env):
        self.obs = observations
        self.env = env


    def match(self):
        mcts = MCTSMatching(self.obs, self.env)
        self.vp_targets, self.passenger_targets = mcts.search()