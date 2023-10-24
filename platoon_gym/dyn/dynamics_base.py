from abc import ABC, abstractmethod
import numpy as np


class DynamicsBase(ABC):
    """
    Base dynamics class.

    :param dt: discrete timestep
    :param x_lims: shape (n, 2), minimum and maximum for each state
    :param u_lims: shape (m, 2), minimum and maximum for each input
    """
    def __init__(self, dt: float, x_lims: np.ndarray, u_lims: np.ndarray):
        self.dt = dt
        self.x_lims = x_lims
        self.u_lims = u_lims
        self.n = None
        self.m = None
        self.p = None
    
    @abstractmethod
    def forward(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Forward dynamics functon. Returns the state at the next timestep when 
        starting at a current state and applying some input.

        :param x: shape (n,), current state
        :param u: shape (m,), input
        :return: state of the vehicle at the next timestep
        """
        pass

    @abstractmethod
    def sense(self, x: np.ndarray) -> np.ndarray:
        """
        Sensing function. Returns an some function of the state that one may 
        have access to due to some sensor.

        :param x: shape (n,), current state
        :return: shape (p,), sensor observation
        """
        pass
    
    def clip_input(self, u: np.ndarray) -> np.ndarray:
        """
        Clips inputs to lie in u_lims.
        
        :param u: input to clip
        :return: shape (m,), clipped input
        """
        return np.clip(u, self.u_lims[:, 0], self.u_lims[:, 1])