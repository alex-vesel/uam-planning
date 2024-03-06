import numpy as np

class eVTOL():
    def __init__(self, id, 
                 max_speed, 
                 max_accel,
                 x_init,
                 y_init,
                ):
        self.id = id
        self.speed = max_speed
        self.max_speed = max_speed
        self.max_accel = max_accel
        self.x = x_init
        self.y = y_init
        self.passenger = None
        self.theta = 0
        self.grounded = False
        self.target = None

        # recording attributes
        self.trip_distance = 0


    def step(self, d_t, action, passenger=None):
        if passenger != self.passenger:
            self.trip_distance = 0
        self.trip_distance += self.speed * d_t
        self.passenger = passenger

        if action.is_flight_action:
            self.theta += action.d_theta
            if action.land:
                self.grounded = True
        else:
            self.theta = action.takeoff_theta
            if not action.stay:
                self.grounded = False
                
        if not self.grounded:
            self.x += self.speed * np.cos(self.theta) * d_t
            self.y += self.speed * np.sin(self.theta) * d_t


    def __repr__(self):
        return f'eVTOL({self.id}, {self.x}, {self.y}, {self.theta})'
    

    def __str__(self):
        return f'eVTOL {self.id} at ({self.x}, {self.y}, {self.theta})'


class eVTOLObservation():
    def __init__(self,
                 agent,
                 vp_distances,
                 vp_headings,
                 vp_num_passengers,
                 vp_radius,
                 agent_distances,
                 agent_headings,
                 is_conflict,
                ):
        self.id = agent.id
        self.x = agent.x
        self.y = agent.y
        self.speed = agent.speed
        self.heading = agent.theta
        self.passenger = agent.passenger
        self.target = agent.target
        self.is_grounded = agent.grounded
        self.vp_distances = vp_distances
        self.vp_headings = vp_headings
        self.vp_num_passengers = vp_num_passengers
        self.vp_radius = vp_radius
        # if self.passenger:
        #     self.can_land = [(d < vp_radius and i == self.passenger.destination) for i, d in enumerate(vp_distances)]
        # else:
        #     self.can_land = [(d < vp_radius and num > 0) for d, num in zip(vp_distances, vp_num_passengers)]
        self.can_land = vp_distances[self.target] < vp_radius
        agent_distances[agent.id] = np.inf
        self.agent_distances = agent_distances
        self.agent_headings = agent_headings
        self.is_conflict = is_conflict

    def update_target(self, new_target):
        self.target = new_target
        self.can_land = self.vp_distances[self.target] < self.vp_radius

