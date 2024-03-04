class eVTOLFlightAction():
    def __init__(self,
                 d_theta,
                 land=False,
                ):
        self.d_theta = d_theta
        self.land = land
        self.is_flight_action = True

    
    def __repr__(self):
        return f'eVTOLFlightAction({self.d_theta}, {self.land})'
    

class eVTOLGroundAction():
    def __init__(self,
                 takeoff_theta,
                 stay=False,
    ):
        self.takeoff_theta = takeoff_theta
        self.land = False
        self.is_flight_action = False
        self.stay = stay


    def __repr__(self):
        return f'eVTOLGroundAction({self.takeoff_theta}, {self.stay})'
    

class eVTOLNullAction():
    def __init__(self):
        pass


    def __repr__(self):
        return f'eVTOLAction(NULL)'