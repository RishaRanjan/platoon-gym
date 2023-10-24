import numpy as np

from platoon_gym.dyn.dynamics_base import DynamicsBase
from platoon_gym.dyn.utils import (DISCRETIZATION_METHODS,
                                  forward_euler_discretization,
                                  pw_const_input_discretization)


class LinearVel(DynamicsBase):
    """
    Simplest discrete-time velocity-based linear model for longitudinal vehicle 
    dynamics with a time delay. In this model, the state and output are given 
    by (position, velocity) and the input is the desired velocity.

    :param dt: discrete timestep
    :param x_lims: shape (n, 2), minimum and maximum for each state
    :param u_lims: shape (m, 2), minimum and maximum for each input
    :param tau: the time delay [s]
    """
    def __init__(self, 
                 dt: float, 
                 x_lims: np.ndarray, 
                 u_lims: np.ndarray, 
                 tau: float,
                 discretization_method = 'forward euler'):
        super().__init__(dt, x_lims, u_lims)

        assert discretization_method in DISCRETIZATION_METHODS,\
            f"discretization method must be one of {DISCRETIZATION_METHODS}"

        # continuous-time dynamics
        Ac = np.array([[0, 1],
                       [0, -1/tau]])
        Bc = np.array([[0],
                       [1/tau]])

        self.n = Ac.shape[0]
        self.m = Bc.shape[1]
        
        # discretization
        if discretization_method == 'forward euler':
            self.Ad, self.Bd = forward_euler_discretization(Ac, Bc, dt)
        elif discretization_method == 'piecewise constant input':
            self.Ad, self.Bd = pw_const_input_discretization(Ac, Bc, dt)

        self.C = np.eye(self.n)
        self.p = self.C.shape[0]

        assert x_lims.shape == (self.n, 2)
        assert u_lims.shape == (self.m, 2)

    def forward(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Forward dynamics functon. Returns the state at the next timestep when 
        starting at a current state and applying some input.

        :param x: shape (n,), current state
        :param u: shape (m,), input
        :return: state of the vehicle at the next timestep
        """
        return self.Ad @ x + self.Bd @ u
    
    def sense(self, x: np.ndarray) -> np.ndarray:
        """
        Sensing function. Returns an some function of the state that one may 
        have access to due to some sensor.

        :param x: shape (n,), current state
        :return: shape (p,), sensor observation
        """
        return self.C @ x