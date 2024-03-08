import numpy as np
from simulator import ms_to_kms

np.random.seed(40)

class Config():
    ## Meta params
    PLOT = False

    ## Simulator parameters
    N_VERTIPORTS = 5
    N_AGENTS = 20
    MAP_SIZE = 120           # km
    D_T = 10              # s
    MAX_TIME = 20000          # s
    MAX_PASSENGERS = 8 * N_AGENTS
    ARRIVAL_RATE_SCALE = 1  # how many times nominal total inflow rate

    # Policy parameters
    POLICY = "greedy"
    MATCHING = "cluster"

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

    avg_trip_distance = 2 * MAP_SIZE / 3
    avg_trip_time = avg_trip_distance / MAX_SPEED_KMS
    evtol_trips_per_hr = 3600 / avg_trip_time
    ARRIVAL_RATE = int(N_AGENTS * evtol_trips_per_hr * ARRIVAL_RATE_SCALE)
    print(ARRIVAL_RATE)
    