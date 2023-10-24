import cvxpy as cp
import numpy as np
from typing import Optional, Tuple

from platoon_gym.ctrl.controller_base import ControllerBase


class LinearMPC(ControllerBase):
    """
    Classic linear MPC implementation using CVXPY and OSQP.

    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        C: Optional[np.ndarray] = None,
        Qf: Optional[np.ndarray] = None,
    ):
        """
        Args:
            A: shape (n, n), state transition matrix
            B: shape (n, m), control matrix
            C: shape (p, n), output matrix
            Q: shape (n, n), state cost matrix
            R: shape (m, m), input cost matrix
            Qf: shape (n, n), terminal state cost matrix
            H: int, prediction horizon
        """
        pass

    def control(self) -> Tuple[np.ndarray, dict]:
        """
        Returns a control input based on a linear feedback control policy. Also
        returns a dict the planned trajectory.

        Args:
            e: shape (n,), error or state (if driving state to zero) vector

        Returns:
            np.ndarray, shape (m,): control input
            dict: empty
        """
        return np.array([]), {}
