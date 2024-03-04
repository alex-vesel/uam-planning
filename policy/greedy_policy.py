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
        # land = False
        # desired_heading = self.obs.heading
        # if self.obs.passenger is not None:
        #     # go to destination if not at destination
        #     if self.obs.vp_distances[self.obs.passenger.destination] > 1:
        #         desired_heading = self.obs.vp_headings[self.obs.passenger.destination]
        #     else:
        #         land = True
        # else:
        #     min_distance = np.inf
        #     min_vertiport = Noned
        #     for i, distance in enumerate(self.obs.vp_distances):
        #         if self.obs.vp_num_passengers[i] > 0 and distance < min_distance:
        #             min_distance = distance
        #             min_vertiport = i

        #     if min_distance < 1:
        #         land = True
        #     else:
        #         if min_vertiport is not None:
        #             desired_heading = self.obs.vp_headings[min_vertiport]
        
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