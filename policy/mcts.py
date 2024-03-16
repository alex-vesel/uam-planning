import numpy as np
import copy
from profilehooks import profile
from .actions import eVTOLFlightAction, eVTOLGroundAction, eVTOLNullAction
from .greedy_policy import GreedyPolicy

FLIGHT_ACTIONS = [
    eVTOLFlightAction(0, False),
    eVTOLFlightAction(-0.4, False),
    # eVTOLFlightAction(-0.1, False),
    # eVTOLFlightAction(0.1, False),
    eVTOLFlightAction(0.4, False),
    eVTOLFlightAction(0, True)
]

GROUND_ACTIONS = [
    eVTOLGroundAction(0, stay=True),
    eVTOLGroundAction(0),
    eVTOLGroundAction(np.pi / 2),
    eVTOLGroundAction(np.pi),
    eVTOLGroundAction(3 * np.pi / 2),
]


class MCTSPolicy:
    def __init__(self, observation, idx, env, all_action=None):
        init_action = all_action[idx] if all_action else None
        init_actions=None
        # if init_action:
        #     # get range around init action
        #     init_actions = [eVTOLFlightAction(init_action.d_theta-0.2, False), init_action, eVTOLFlightAction(init_action.d_theta+0.2, False)]
        #     if np.any(observation.can_land):
        #         init_actions.append(eVTOLFlightAction(0, True))

        self.init_action = init_action
        init_state = MCTSState(observation, idx, env, init_actions=init_actions)
        self.root = MCTSNode(init_state, all_action=all_action)
        self.idx = idx
        self.simulations = 50#100
        self.search_depth = 4

    # @profile
    def search(self, return_list=None):
        # if agent is grounded and at target, return null action
        if self.root.state.obs.is_grounded and np.argmin(self.root.state.obs.vp_distances) == self.root.state.obs.target:
            return eVTOLGroundAction(0, stay=True)
        
        # if agent not within 5 units of other agents, return greedy policy
        if np.min(self.root.state.obs.agent_distances) > 5:
            return GreedyPolicy(self.root.state.obs, self.idx, self.root.state.env).search()

        for _ in range(self.simulations):
            v = self.tree_policy()
            reward = v.rollout(self.search_depth)
            v.backpropagate(reward)

        best_action = self.root.best_child(c_param=0.).state.prev_action

        if return_list is not None:
            return_list[self.idx] = best_action

        return best_action


    def tree_policy(self):
        current_node = self.root
        while not current_node.is_terminal_node(self.search_depth):
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()

        return current_node
    

class MCTSState:
    def __init__(self, observation, idx, env, prev_action=eVTOLFlightAction(0, False), depth=0, init_actions=None):
        self.obs = observation
        self.idx = idx
        self.env = env
        self.prev_action = prev_action
        self.depth = depth
        self.init_actions = init_actions


    def reward(self):
        dist_x = self.obs.x - self.env.map.vertiports[self.obs.target].x
        dist_y = self.obs.y - self.env.map.vertiports[self.obs.target].y

        target_dist = np.sqrt(dist_x ** 2 + dist_y ** 2)
        if self.obs.is_grounded and target_dist < 1:
            return 100
        
        if self.obs.is_conflict:
            return -100
        
        return 1 / (target_dist + 1)


    def get_legal_actions(self):
        if self.obs.is_grounded:
            return copy.copy(GROUND_ACTIONS)
        elif self.init_actions is not None:
            return self.init_actions
        else:
            if self.obs.can_land:
                return copy.copy(FLIGHT_ACTIONS)
            else:
                return copy.copy(FLIGHT_ACTIONS)[:-1]
        

    def step(self, actions, rollout=False):
        # copy environment to avoid modifying the original
        if rollout:
            new_env = self.env
        else:
            new_env = copy.deepcopy(self.env)
        obs = new_env.step(actions, step_map=False)[self.idx]
        return MCTSState(obs, self.idx, new_env, prev_action=actions[self.idx], depth=self.depth + 1)
    

    def is_terminal_state(self, max_depth):
        return self.depth >= max_depth or self.obs.is_conflict# or self.obs.is_grounded# or self.obs.delivered_passenger or self.obs.has_new_passenger
    

class MCTSNode:
    def __init__(self, state, all_action=None, parent=None):
        self.state = state
        self.all_action = all_action
        if not all_action:
            all_action_idx = np.random.randint(0, len(FLIGHT_ACTIONS), size=self.state.obs.agent_distances.shape[0])
            self.all_action = [FLIGHT_ACTIONS[i] for i in all_action_idx]
        self.parent = parent
        self.children = []
        self.q = 0
        self.n = 0
        self.untried_actions = state.get_legal_actions()


    def reward(self):
        return self.q / self.n if self.n != 0 else 0
    

    def expand(self):
        a = self.untried_actions.pop()
        self.all_action[self.state.idx] = a
        next_state = self.state.step(self.all_action)
        child_node = MCTSNode(next_state, parent=self)
        self.children.append(child_node)
        return child_node
    

    def rollout(self, max_depth):
        rollout_state = self.state
        rollout_flag = False
        while not rollout_state.is_terminal_state(max_depth):
            # all_action_idx = np.random.randint(0, len(FLIGHT_ACTIONS), size=self.state.obs.agent_distances.shape[0])
            # all_action = [FLIGHT_ACTIONS[i] for i in all_action_idx]

            # initialize all_action with the best action from the current state
            all_action = [None for _ in range(self.state.env.config.N_AGENTS)]
            for i, observation in enumerate(self.state.env.observations):
                all_action[i] = GreedyPolicy(observation, i, self.state.env).search()
            rollout_state = rollout_state.step(all_action, rollout=rollout_flag)
            rollout_flag = True
        return rollout_state.reward()
    

    def backpropagate(self, reward):
        lookahead_reward = self.state.reward() + 0.99 * reward
        self.n += 1
        self.q += lookahead_reward
        if self.parent:
            self.parent.backpropagate(lookahead_reward)


    def best_child(self, c_param=1.4): 
        if len(self.children) == 0:
            return self
        choices_weights = [
            c.reward() + c_param * np.sqrt((2 * np.log(self.n) / c.n))
            for c in self.children
        ]
        best_indices = np.flatnonzero(choices_weights == np.max(choices_weights))
        return self.children[np.random.choice(best_indices)]



    def is_terminal_node(self, max_depth):
        return self.state.is_terminal_state(max_depth)


    def is_fully_expanded(self):
        return len(self.untried_actions) == 0
    

    def __repr__(self):
        return f'MCTSNode({self.state.obs}, {self.q}, {self.n}, {self.reward()})'