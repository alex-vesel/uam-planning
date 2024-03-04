import numpy as np
from simulator import ms_to_kms

np.random.seed(10)

class Config():
    ## Meta params
    PLOT = False

    ## Simulator parameters
    N_VERTIPORTS = 10
    N_AGENTS = 40
    MAP_SIZE = 80           # km
    D_T = 10              # s
    MAX_TIME = 5000          # s
    MAX_PASSENGERS = 100

    # Policy parameters
    POLICY = "mcts"
    MATCHING = "greedy"

    ## Safety parameters
    EVENT_COOLDOWN = 60       # s
    LOS_DIST = 0.926          # km
    NMAC_DIST = 0.150         # km 

    ## eVTOL parameters
    MAX_SPEED_MS = 90          # m/s
    MAX_ACCEL_MS = 10          # m/s^2

    ## Vertiport parameters
    VERTIPORT_RADIUS = 1700  # m

    ## Calculations
    MAX_SPEED_KMS = ms_to_kms(MAX_SPEED_MS)
    MAX_ACCEL_KMS = ms_to_kms(MAX_ACCEL_MS)
    VERTIPORT_RADIUS_KM = VERTIPORT_RADIUS / 1000
    