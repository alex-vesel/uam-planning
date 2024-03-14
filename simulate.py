import numpy as np
import copy
from tqdm import tqdm
from time import sleep
from time import time

from config import Config
from simulator import *
from policy import *
from matching import *

# Questions:
# 1. If a NMAC, should the agent be removed from the map?
    # no
# 2. How should LOS and NMAC cooldown be handled?
    # that's fine
# 5. How to do gobal reward in a scenario where each agent cannot plan knowing the state of every other agent?
    # assume agents know reward, centralized approach is fine
# 6. Currently every eVTOL knows the position of every other aircraft. Is this a realistic assumption?
    # yes, ADS-B

# Central problem:
# Developing a decentralized/partially centralized multiagent (100+ agents) planner that achieves 
# long-term objectives (passenger throughput) when satisfying short-term constraints 
# (avoiding collisions) in problems where short-term constraint satisfication will affect 
# long-term objectives. 

# Potential solutions
#   - Hiererchical model
#   - Centralized model for planning, decentralized for execution

# Mykel's feedback
# Centralized approach is principled, communications would allow for trajectories to be broadcast
#   ADS-B will allow for positions and velocities to be known
# consideration: 2 evtol copmanies, each centrally optimizing but needing to be responsible for avoiding
#     collisions with the other company


# idea:
# - every iteration divide airspace into sectors for collision avoidance using clustering algorithm to parallelize
# - multiscale approach, plan at high temporal resolution, finetune at low temporal resolution
# - for high level planning, view passengers and destinations as a directed graph?
    # - high level plan: assign each agent a passenger (uncertainty in future passengers)
        # - evaluating candidate solutions: lookahead to future state given a current assignment
            # - assuming optimal route following, where are all agents? run value estimate on future state
            # - run MCTS on discrete steps into future given assignment, generate average score over many simulations
            # - optimize initial assignment to maximize future value
    # - at low level, introduce noise to actions (uncertainty in aircraft dynamics)

# IDEA:
# Two scales of planning:
#   - high level: matching passengers to agents
#   - low level: planning agent trajectories

# todo:
# initialize all action index for each agent with greedy policy instead of random
#   especially in MCTS
# allow agent to land anywhere, pick up passenger as something that is given to a queue of agents at a vertiport
# run second level of MCTS to produce more granular actions


# possible ablations
# In MCTS, other agents greedy vs random actions
# 2 round of MCTS with second round doing granular actions (or expanded action set for MCTS root)
# Matching algorithms
#   Greedy (closest) vertiport
#   Greedy (closest) passenger
#   Current state hungarian matching
# Flight algorithms
#   Greedy (take action that gets you closest to target) ignoring collisions
# add to metrics percent flight time occupied by passengers


# high level planner
# - currently takes into account the current passenger layout
#  - costs are total passenger waiting time by the time an agent arrives
#  - however, a better planner should take into account
#    1. current passengers
#   2. future passengers   
#   3. congestion, should not assign more agents to a "sector" than necessary


# CURRENT TODO
# do alternating high level then low level planning
# generate candidate solutions using heuristic, then rollout
    # perhaps rollout in a future state?
# if passenger is going to high value vertiport account for that in score
# in vertiport scores, consider if area is already covered by other agents
# account that if close to delviering to a vertiport, agent with passenger can be assigned
#   to a new passenger
    # however, this logic must be thoguth through carefully, agents must be assigned, so 
    # delvering agent can "swipe" from other nearby agents because we must assign next passenger,
    # even if not optimal
# consider sum of all agent reward in MCTS, allows agents to behave more cooperatively rather than
    # iteratively greedy

def simulate(env, policy, matching, plot=True):
    joint_observations = env.reset()

    all_action_idx = np.random.randint(0, len(FLIGHT_ACTIONS), size=env.config.N_AGENTS)
    all_action_orig = [FLIGHT_ACTIONS[i] for i in all_action_idx]

    while not env.done():
        # print(env.passengers_served)
        # get target destination for each agent
        # matching = GreedyPassengerMatching(joint_observations, env)

        matching_obj = matching(joint_observations, env)
        matching_obj.match()
        for i, agent in enumerate(env.agents):
            agent.target = matching_obj.vp_targets[i]
            agent.passenger_target = matching_obj.passenger_targets[i]
            # update observation target
            joint_observations[i].update_target(matching_obj.vp_targets[i])

        all_action = [None for _ in range(env.config.N_AGENTS)]
        for i, observation in enumerate(joint_observations):
            all_action[i] = GreedyPolicy(observation, i, env).search()

        # prev_matching = matching_obj.vp_targets

        # print(matching_obj.vp_targets)

        # all_action = None

        # score = MatchingScore(joint_observations, copy.deepcopy(env)).score()

        # get action for each agent
        actions = []
        for i, observation in enumerate(joint_observations):
            action = policy(observation, i, copy.deepcopy(env), all_action=all_action).search()
            all_action[i] = action
            actions.append(action)

        # execute action and step environment
        joint_observations = env.step(actions, verbose=False)
        if plot:
            env.plot()

    env.finish()

    # Print statistics
    print("Time: ", env.time)
    print("Passengers served: ", env.passengers_served)
    print(f'NMACs: {env.NMAC_events}')
    print(f'LOSs: {env.LOS_events}')
    print(f'NMACs/h: {env.NMAC_events_h}')
    print(f'LOSs/h: {env.LOS_events_h}')
    print("Trip ratio: ", env.trip_ratio)
    print("Passengers/h: ", env.passengers_h)
    print("Average passenger wait time: ", env.avg_wait_time)
    print("Max passenger wait time: ", env.max_wait_time)


def run_experiment(config):
    if config.MAP_TYPE == "random":
        map = RandomMap(
            config
        )
    else:
        map = SatMap(
            config
        )

    env = Environment(
        map=map,
        config=config
    )

    if config.POLICY == "mcts":
        policy = MCTSPolicy
    else:
        policy = GreedyPolicy

    if config.MATCHING == "greedy":
        matching = GreedyPassengerMatching
    elif config.MATCHING == "hungarian":
        matching = HungarianMatching
    elif config.MATCHING == "lookahead":
        matching = LookaheadMatching
    else:
        matching = MCTSHungarianMatching

    start = time()
    simulate(env, policy, matching, plot=config.PLOT)
    end = time()
    print("Runtime: ", end - start)

    return env


if __name__ == "__main__":
    run_experiment(Config)