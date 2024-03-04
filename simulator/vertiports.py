import numpy as np

from .utils import per_hr_to_per_s

class Vertiport:
    def __init__(self, id, x, y, dt, num_vertiports, arrival_rate):
        self.id = id
        self.x = x
        self.y = y
        self.dt = dt
        self.time = 0
        self.total_passengers = 0
        self.num_vertiports = num_vertiports
        self.passengers = []
        self.landed_agents = []
        self.arrival_rate = arrival_rate


    def step(self, done_generating=False):
        if done_generating:
            num_new_passengers = 0
        else:
            num_new_passengers = np.random.poisson(per_hr_to_per_s(self.arrival_rate)*self.dt)
        self.total_passengers += num_new_passengers
        for _ in range(num_new_passengers):
            origin = self.id
            destination = np.random.randint(self.num_vertiports)
            while destination == origin:
                destination = np.random.randint(self.num_vertiports)
            self.passengers.append(Passenger(len(self.passengers), self.time, origin, destination))

        matches = []
        while len(self.passengers) > 0 and len(self.landed_agents) > 0:
            passenger = self.passengers.pop(0)
            agent = self.landed_agents.pop(0)
            matches.append((agent.id, passenger))
        self.landed_agents = []

        self.time += self.dt

        return matches


    def land(self, agent):
        if agent.passenger and agent.passenger.destination == self.id:
            return True
        return False
    

    def ground(self, agent):
        if agent not in self.landed_agents:
            self.landed_agents.append(agent)
    

    def get_waiting_times(self):
        return [self.time - passenger.start_time for passenger in self.passengers]

    
    def __repr__(self):
        return f'Vertiport({self.id}, {self.x}, {self.y})'
    

class Passenger:
    def __init__(self, id, time, origin, destination):
        self.id = id
        self.start_time = time
        self.origin = origin
        self.destination = destination