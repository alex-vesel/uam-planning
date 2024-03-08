import numpy as np
from collections import defaultdict

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
        self.cur_passengers = []
        self.landed_agents = []
        self.arrival_rate_h = arrival_rate
        self.arrival_rate_s = per_hr_to_per_s(arrival_rate)
        self.loading = []
        self.passenger_arrivals = defaultdict(list)


    def generate_passengers(self):
        num_new_passengers = np.random.poisson(self.arrival_rate_s*self.dt)
        for _ in range(num_new_passengers):
            self.total_passengers += 1
            origin = self.id
            destination = np.random.randint(self.num_vertiports)
            while destination == origin:
                destination = np.random.randint(self.num_vertiports)
            self.passenger_arrivals[self.time].append(Passenger(self.total_passengers, self.time, origin, destination))
        self.time += self.dt


    def step(self, done_generating=False):
        for passenger in self.passenger_arrivals[self.time]:
            self.total_passengers += 1
            self.cur_passengers.append(passenger)
        # add 60 second loading time
        # add fixed number of landing strips
        matches = []
        while len(self.cur_passengers) > 0 and len(self.landed_agents) > 0:
            if self.landed_agents[0].passenger_target and self.landed_agents[0].passenger_target in self.cur_passengers:
                passenger = self.landed_agents[0].passenger_target
                self.cur_passengers.remove(passenger)
                agent = self.landed_agents.pop(0)
                matches.append((agent.id, passenger))
            else:
                passenger = self.cur_passengers.pop(0)
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
        return [self.time - passenger.start_time for passenger in self.cur_passengers]
    

    def update_waiting_times(self):
        self.waiting_times = self.get_waiting_times()
        for i, passenger in enumerate(self.cur_passengers):
            passenger.wait_time = self.waiting_times[i]
    

    def get_passengers(self):
        self.update_waiting_times()
        return self.cur_passengers

    
    def __repr__(self):
        return f'Vertiport({self.id}, {self.x}, {self.y})'
    

class Passenger:
    def __init__(self, id, time, origin, destination):
        self.id = id
        self.start_time = time
        self.origin = origin
        self.destination = destination
        self.wait_time = 0