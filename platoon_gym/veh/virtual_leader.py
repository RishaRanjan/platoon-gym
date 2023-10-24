"""Virtual leader class that provides a reference trajectory for platoon."""

import numpy as np

from platoon_gym.veh.utils import VL_TRAJECTORY_TYPES


class VirtualLeader:
    """
    Virtual leader that (at least) the platoon leader has access to.

    Attributes:
       state: np.ndarray, state (p, v, a) of the virtual leader
    """

    def __init__(
        self,
        trajectory_type: str,
        trajectory_args: dict,
        position: float = 0.0,
        velocity: float = 0.0,
        acceleration: float = 0.0,
    ):
        """
        Initialize the virtual leader with its trajectory type and initial state.

        Args:
            trajectory_type: str, type of trajectory to follow
            trajectory_args: dict, arguments for the trajectory
            position: float, initial position of the virtual leader
            velocity: float, initial velocity of the virtual leader
            acceleration: float, initial acceleration of the virtual leader
        """
        assert trajectory_type in VL_TRAJECTORY_TYPES
        self.traj_type = trajectory_type
        self.state = np.array([position, velocity, acceleration])
        self.traj_args = trajectory_args
        self.H = self.traj_args["horizon"]
        self.dt = self.traj_args["dt"]
        self.n = self.state.shape[0]
        self.plan = np.empty((self.n, self.H + 1))
        self.time = 0.0
        self.time_forecast = np.arange(self.H + 1) * self.dt
        self.init_traj()

    def init_traj(self):
        """Initialize the trajectory."""
        if self.traj_type == "constant velocity":
            assert self.state[0] == 0.0, "Initial position must be 0.0"
            assert self.state[2] == 0.0, "Initial acceleration must be 0.0"
            self.plan[0, :] = self.state[0] + self.state[1] * self.time_forecast
            self.plan[1, :] = self.state[1]
            self.plan[2, :] = 0.0
        else:
            raise NotImplementedError

    def step(self):
        """Step the virtual leader by one time step."""
        if self.traj_type == "constant velocity":
            self.state[0] += self.state[1] * self.dt
            self.time += self.dt
            self.time_forecast += self.dt
            self.plan[:, :-1] = self.plan[:, 1:]
            self.plan[0, -1] = self.plan[0, -2] + self.dt * self.plan[1, -2]
            self.plan[1, -1] = self.plan[1, -2]
            self.plan[2, -1] = 0.0
        else:
            raise NotImplementedError
