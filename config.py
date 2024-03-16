import numpy as np
from simulator import ms_to_kms

np.random.seed(2424)

class Config():
    ## Meta params
    PLOT = False

    ## Simulator parameters
    N_VERTIPORTS = 10
    N_AGENTS = 40
    MAP_SIZE = 120           # km
    D_T = 10              # s
    MAX_TIME = 20000          # s
    MAX_PASSENGERS = 5 * N_AGENTS
    ARRIVAL_RATE_SCALE = 1  # how many times nominal total inflow rate
    MAP_TYPE = "sf"

    # Policy parameters
    # POLICY = "greedy"
    POLICY = "greedy"
    MATCHING = 'lookahead'
    # MATCHING = "cluster"
    # MATCHING = "hungarian"
    # MATCHING = "greedy"

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
    if MAP_TYPE == "sf":
        MAP_SIZE = 120
    elif MAP_TYPE == "nyc":
        MAP_SIZE = 40

    MAX_SPEED_KMS = ms_to_kms(MAX_SPEED_MS)
    MAX_ACCEL_KMS = ms_to_kms(MAX_ACCEL_MS)
    VERTIPORT_RADIUS_KM = VERTIPORT_RADIUS / 1000

    avg_trip_distance = 2 * MAP_SIZE / 3
    avg_trip_time = avg_trip_distance / MAX_SPEED_KMS
    evtol_trips_per_hr = 3600 / avg_trip_time
    ARRIVAL_RATE = int(N_AGENTS * evtol_trips_per_hr * ARRIVAL_RATE_SCALE)
    print("Network passenger arrival rate: ", ARRIVAL_RATE)


def config_calculations(config):
    if config.MAP_TYPE == "sf":
        config.MAP_SIZE = 120
    elif config.MAP_TYPE == "nyc":
        config.MAP_SIZE = 40

    config.MAX_SPEED_KMS = ms_to_kms(config.MAX_SPEED_MS)
    config.MAX_ACCEL_KMS = ms_to_kms(config.MAX_ACCEL_MS)
    config.VERTIPORT_RADIUS_KM = config.VERTIPORT_RADIUS / 1000

    avg_trip_distance = 2 * config.MAP_SIZE / 3
    avg_trip_time = avg_trip_distance / config.MAX_SPEED_KMS
    evtol_trips_per_hr = 3600 / avg_trip_time
    config.ARRIVAL_RATE = int(config.N_AGENTS * evtol_trips_per_hr * config.ARRIVAL_RATE_SCALE)
    print("Network passenger arrival rate: ", config.ARRIVAL_RATE)


if __name__ == "__main__":
    config = Config()