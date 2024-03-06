import numpy as np
from .actions import eVTOLFlightAction, eVTOLGroundAction, eVTOLNullAction


class GreedyPolicy():
    def __init__(self, observation, idx, env, all_action=None):
        self.obs = observation
        self.idx = idx
        self.env = env

    def search(self):
        # find closest vertiport with passengers
        desired_heading = self.obs.vp_headings[self.obs.target]

        # if agent is grounded and at target, return stay action
        if self.obs.is_grounded and self.obs.target == np.argmin(self.obs.vp_distances):
            return eVTOLGroundAction(0, stay=True)
        
        if self.obs.is_grounded:
            return eVTOLGroundAction(desired_heading, stay=False)

        # if target is close land
        if self.obs.can_land:
            return eVTOLFlightAction(0, True)
        
        # get closest circular direction to desired heading
        d_theta = (desired_heading - self.obs.heading) % (2 * np.pi)
        if d_theta > np.pi:
            d_theta -= 2 * np.pi
        if d_theta < -np.pi:
            d_theta += 2 * np.pi

        if d_theta > 0:
            d_theta = min(0.4, d_theta)
        else:
            d_theta = max(-0.4, d_theta)

        return eVTOLFlightAction(d_theta, False)