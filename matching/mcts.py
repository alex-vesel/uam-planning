import numpy as np
import copy
from profilehooks import profile
from policy.greedy_policy import GreedyPolicy
from .hungarian_matching import HungarianMatching


class MCTSMatching:
    def __init__(self, observation, env):
        init_state = MCTSState(observation, env, prev_passengers_served=env.passengers_served)
        self.root = MCTSNode(init_state, num_actions=2)
        self.simulations = 2#100
        self.search_depth = 1

    # @profile
    def search(self, return_list=None):
        for _ in range(self.simulations):
            v = self.tree_policy()
            reward = v.rollout(self.search_depth)
            vp_targets = self.root.children[0].all_vp_targets[-1, :].tolist()
            for i, agent in enumerate(self.root.state.obs):
                if agent.passenger:
                    vp_targets[i] = agent.passenger.destination
            return (vp_targets, self.root.best_child(c_param=0.).state.prev_action[1])
            v.backpropagate(reward)
            

        # print rewards of all children
        # print(self.root.state.env.time)
        # rewards = [child.reward() for child in self.root.children]
        # print(rewards)
        # if len(rewards) > 1:  
        #     for child in self.root.children:
        #         print(child.state.prev_action)
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
    def __init__(self, observation, env, prev_action=None, depth=0, prev_passengers_served=0):
        self.obs = observation
        self.env = env
        self.depth = depth
        self.prev_action = prev_action
        self.prev_passengers_served = prev_passengers_served
        self.match_obj = HungarianMatching(self.obs, self.env)



    def reward(self):
        # get average distance to target for all agents
        # return 0
        if self.obs[0].target is None:
            return 0

        distances = np.zeros(len(self.obs))
        for i, obs in enumerate(self.obs):
            dist_x = obs.x - self.env.map.vertiports[obs.target].x
            dist_y = obs.y - self.env.map.vertiports[obs.target].y
            distances[i] = np.sqrt(dist_x ** 2 + dist_y ** 2)
        avg_distance = np.mean(distances)
        return 1 / (avg_distance + 1)

        new_passengers_served = self.env.passengers_served - self.prev_passengers_served

        return new_passengers_served

        # return 1 / (avg_distance + 1) + new_passengers_served

        # dist_x = self.obs.x - self.env.map.vertiports[self.obs.target].x
        # dist_y = self.obs.y - self.env.map.vertiports[self.obs.target].y

        # target_dist = np.sqrt(dist_x ** 2 + dist_y ** 2)
        # if self.obs.is_grounded and target_dist < 1:
        #     return 100
        
        # if self.obs.is_conflict:
        #     return -100
        

        # return 1 / (target_dist + 1) + turn_penalty


    def get_actions(self, num_actions=1):
        actions = []
        for _ in range(num_actions):
            if self.match_obj.ok:
                vp_targets, passenger_targets = self.match_obj.match()
                if vp_targets is None:
                    continue
                actions.append((vp_targets, passenger_targets))
        return actions
        

    def step(self, vp_targets, passenger_targets, rollout=False):
        # copy environment to avoid modifying the original
        if rollout:
            new_env = self.env
        else:
            new_env = copy.deepcopy(self.env)
            new_env.dt = 10

        obs = copy.deepcopy(self.obs)
        for _ in range(1):
            actions = []
            for i, agent in enumerate(new_env.agents):
                agent.target = vp_targets[i]
                agent.passenger_target = passenger_targets[i]
                obs[i].update_target(vp_targets[i])
                actions.append(GreedyPolicy(obs[i], i, new_env).search())
            obs = new_env.step(actions, step_map=True)
        return MCTSState(obs, new_env, prev_action=(vp_targets, passenger_targets), depth=self.depth + 1, prev_passengers_served=self.env.passengers_served)
    

    def is_terminal_state(self, max_depth):
        # return self.depth >= max_depth
        return self.depth >= 2
    

class MCTSNode:
    def __init__(self, state, num_actions=1, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.q = 0
        self.n = 0
        self.untried_actions = self.state.get_actions(num_actions)


    def reward(self):
        return self.q / self.n if self.n != 0 else 0
    

    def expand(self):
        vp_targets, passenger_targets = self.untried_actions.pop()
        next_state = self.state.step(vp_targets, passenger_targets)
        child_node = MCTSNode(next_state, parent=self)
        self.children.append(child_node)
        return child_node
    

    def rollout(self, max_depth):
        rollout_state = MCTSState(self.state.obs, self.state.env, prev_action=self.state.prev_action, depth=self.state.depth, prev_passengers_served=self.state.prev_passengers_served)
        prev_passengers = rollout_state.env.passengers_served
        passengers_per_time = []
        # rollout_flag = False
        all_vp_targets = []
        while not rollout_state.is_terminal_state(max_depth):
            actions = rollout_state.get_actions()
            vp_targets, passenger_targets = actions[0]
            all_vp_targets.append(vp_targets)
            rollout_state = rollout_state.step(vp_targets, passenger_targets, rollout=False)
            passengers_per_time.append(rollout_state.env.passengers_served - prev_passengers  + 10 * rollout_state.reward())
            prev_passengers = rollout_state.env.passengers_served

        # rowwise number of diffs for each vp_targets
        diffs = []
        for i in range(len(all_vp_targets[0])):
            diffs.append(np.sum([all_vp_targets[j][i] != all_vp_targets[j-1][i] for j in range(1, len(all_vp_targets))]))
        # print(diffs)      
        reward = 0
        for i, p in enumerate(passengers_per_time):
            reward += (p) * (0.95 ** i)
        # print(np.sum(diffs))
        # print(np.array(all_vp_targets))
        self.all_vp_targets = np.array(all_vp_targets)
        reward -= np.sum(diffs)
        return reward
        
        
            # rollout_flag = True
        # for _ in range(5):
        #     all_action_idx = np.random.randint(0, len(FLIGHT_ACTIONS), size=self.state.obs.agent_distances.shape[0])
        #     all_action = [FLIGHT_ACTIONS[i] for i in all_action_idx]

        #     # initialize all_action with the best action from the current state
        #     all_action = [None for _ in range(self.state.env.config.N_AGENTS)]
        #     for i, observation in enumerate(self.state.env.observations):
        #         all_action[i] = GreedyPolicy(observation, i, self.state.env).search()
        #     rollout_state = rollout_state.step(all_action, rollout=rollout_flag)
            # actions = rollout_state.get_actions()
            # vp_targets, passenger_targets = actions[0]
            # rollout_state = rollout_state.step(vp_targets, passenger_targets, rollout=rollout_flag)
            # rollout_flag = True
        # print(rollout_state.env.passengers_served)
        # return rollout_state.reward()
    

    def backpropagate(self, reward):
        lookahead_reward = self.state.reward() + 1 * reward
        self.n += 1
        self.q += lookahead_reward
        if self.parent:
            self.parent.backpropagate(lookahead_reward)


    def best_child(self, c_param=1.4): 
        if len(self.children) == 0:
            return self
        choices_weights = [
            c.reward() + c_param * np.sqrt((2 * np.log(self.n) / c.n)) + 0.01 * (len(self.children) - i)
            for i, c in enumerate(self.children)
        ]
        choices_weights = [1]
        best_indices = np.flatnonzero(choices_weights == np.max(choices_weights))
        return self.children[np.random.choice(best_indices)]


    def is_terminal_node(self, max_depth):
        return self.state.depth >= max_depth
        # return self.state.is_terminal_state(max_depth)


    def is_fully_expanded(self):
        return len(self.untried_actions) == 0
    

    def __repr__(self):
        return f'MCTSNode({self.state.obs}, {self.q}, {self.n}, {self.reward()})'