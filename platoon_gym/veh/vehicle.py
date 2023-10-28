"""Vehicle class for platoon_gym environment.

This class contains all vehicle information needed from the platooning 
environment."""

from typing import Optional, Tuple
import numpy as np

from platoon_gym.dyn.dynamics_base import DynamicsBase


class Vehicle:
    """
    Vehicle class. Contains a dynamics model and the vehicle's state/output.

    Attributes:
        dyn: Derived[DynamicsBase] dynamics class derived from DynamicsBase
        position: float, longitudinal position on road [m]
        velocity: float, longitudinal velocity (signed) [m/s]
        acceleration: float, longitudinal acceleration (signed) [m/s^2],
            optional
    """

    def __init__(
        self,
        dyn: DynamicsBase,
        position: float = 0.0,
        velocity: float = 0.0,
        acceleration: Optional[float] = None,
    ):
        """
        Initialize the vehicle with its dynamics model and starting state.
        """
        self.dyn = dyn
        self.dt = self.dyn.dt
        self.n, self.m, self.p = dyn.n, dyn.m, dyn.p
        if acceleration is not None:
            self.state = np.array([position, velocity, acceleration])
        else:
            self.state = np.array([position, velocity])
        self.output = np.array([position, velocity])

    def step(self, control: np.ndarray):
        """
        Step the vehicle dynamics forward one time step based on the
        control input given by control.

        Args:
            control: np.ndarray, control input to the vehicle dynamics model
        """
        self.state = self.dyn.forward(self.state, control)
        self.output = self.dyn.sense(self.state)
