import os

def get_exp_dir(config):
    return os.path.join(
        "output",
        f"flight_levels_{config.N_FLIGHT_LEVELS}",
        f"policy_{config.POLICY}",
        f"matching_{config.MATCHING}",
        f"map_type_{config.MAP_TYPE}",
        f"agents_{config.N_AGENTS}",
        f"vertiports_{config.N_VERTIPORTS}",
    )
