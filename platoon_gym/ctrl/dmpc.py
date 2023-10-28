import cvxpy as cp
import numpy as np
import scipy as sp
from scipy.linalg import block_diag
from typing import List, Optional, Tuple

from platoon_gym.ctrl.controller_base import ControllerBase


class DistributedMPC(ControllerBase):
    """
    Distributed MPC controller for platoon of vehicles.
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        Q: np.ndarray,
        Qn: np.ndarray,
        R: np.ndarray,
        H: int,
    ):
        """
        Args:
            A: shape (n, n), state transition matrix
            B: shape (n, m), control matrix
            C: shape (p, n), output matrix
            Q: shape (n, n), assumed state cost matrix
            Qn: shape (n, n), neighbor state cost matrix
            R: shape (m, m), input cost matrix
            H: int, prediction horizon
        """
        self.n, self.m = A.shape[0], B.shape[1]
        self.H = H
        pass

    def control(
        self,
        x0: np.ndarray,
        xa: np.ndarray,
        xn: List[np.ndarray],
        u_ref: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Returns a control input based on a distributed MPC policy. Also returns
        a dict with the planned trajectory and planned inputs.

        Args:
            x0: shape (n,), current state
            xa: shape (n, H+1), assumed state trajectory
            xn: list[np.ndarray, shape (n, H+1)], neighbors state trajectories
            u_ref: shape (m, H), optional reference control input

        Returns:
            np.ndarray, shape (m,): control input
            dict: extra information related to the problem
        """
        return np.empty([]), {}

    def mpc_problem(self, x0, z_ref, u_ref) -> Tuple[cp.Problem, cp.Variable]:
        """
        Creates the CVXPY problem and optimization variable for the MPC problem.
        """
        n, m, H = self.n, self.m, self.H
        z = cp.Variable(H * (m + n))
        cost = 0.0
        constraints = []
        return cp.Problem(cp.Minimize(cost), constraints), z
