import cvxpy as cp
import numpy as np

from platoon_gym.ctrl.controller_base import ControllerBase


class LinearMPC(ControllerBase):
    """
    Classic linear MPC implementation using CVXPY and OSQP.

    :param A: shape (n, n), state transition matrix
    :param B: shape (n, m), control matrix
    :param C: shape (p, n), output matrix
    :param Q: shape (n, n), state cost matrix
    :param R: shape (m, m), input cost matrix
    :param Qf: shape (n, n), terminal state cost matrix
    :param H: int, prediction horizon
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        Qf: np.ndarray,
    ):
        pass

    def control(self):
        pass
