from typing import Optional, Type

import numpy as np

from platoon_gym.dyn.dynamics_base import DynamicsBase


class Vehicle:
    """
    Vehicle class. Vehicles have dynamics and use a controller. 

    :param dyn: dynamics class derived from DynamicsBase
    :param position: longitudinal position on road [m]
    :param velocity: longitudinal velocity (signed) [m/s]
    :param acceleration: longitudinal acceleration (signed) [m/s^2], optional
    """
    def __init__(self, 
                 dyn: Type[DynamicsBase], 
                 position: float = 0., 
                 velocity: float = 0., 
                 acceleration: Optional[float] = None):
        self.dyn = dyn
        self.n, self.m, self.p = self.dyn.n, self.dyn.m, self.dyn.p
        if acceleration is not None:
            self.state = np.array([position, velocity, acceleration])
        else:
            self.state = np.array([position, velocity])
        self.output = np.array([position, velocity])
    
    def step(self, input):
        self.state = self.dyn.forward(self.state, input)
        self.output = self.dyn.sense(self.state)