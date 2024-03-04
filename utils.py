import os

def get_exp_dir(config):
    return os.path.join(
        f"policy_{config.POLICY}",
        f"matching_{config.MATCHING}",
        f"vertiports_{config.N_VERTIPORTS}",
        f"agents_{config.N_AGENTS}",
        f"map_{config.MAP_SIZE}",
        f"dt_{config.D_T}",
        f"max_passengers_{config.MAX_PASSENGERS}",
    )
