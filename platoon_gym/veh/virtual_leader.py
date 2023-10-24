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
        """
        assert trajectory_type in VL_TRAJECTORY_TYPES
        self.traj_type = trajectory_type
        self.state = np.array([position, velocity, acceleration])
