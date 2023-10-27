import cvxpy as cp
import numpy as np
import scipy as sp
from scipy.linalg import block_diag
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
        C: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        Qf: np.ndarray,
        Cx: np.ndarray,
        Cu: np.ndarray,
        dx: np.ndarray,
        du: np.ndarray,
        H: int,
    ):
        """
        Args:
            A: shape (n, n), state transition matrix
            B: shape (n, m), control matrix
            C: shape (p, n), output matrix
            Q: shape (n, n), state cost matrix
            R: shape (m, m), input cost matrix
            Qf: shape (n, n), terminal state cost matrix
            Cx: shape (n, n), state constraint matrix
            Cu: shape (m, m), input constraint matrix
            dx: shape (n,), state constraint vector
            du: shape (m,), input constraint vector
            H: int, prediction horizon
        """
        assert H > 1
        n, m, p = A.shape[0], B.shape[1], C.shape[0]
        self.H = H
        self.opt_var_dim = H * (n + m)

        P = block_diag(*([R, C.T @ Q @ C] * (H - 1) + [R, C.T @ Qf @ C]))
        self.P = sp.sparse.csr_matrix(P)
        assert self.P.shape == (self.opt_var_dim, self.opt_var_dim)

        A_bar = np.zeros((H * n, self.opt_var_dim))
        A_bar[:, :-n] += block_diag(*([B] + [np.block([-A, -B])] * (H - 1)))
        A_bar[:-n, m:-n] += block_diag(
            *([np.block([np.eye(n), np.zeros((n, m))])] * (H - 1))
        )
        A_bar[-n:, -n:] += np.eye(n)
        self.A_bar = sp.sparse.csr_matrix(A_bar)

        C_bar = block_diag(*([Cu, Cx] * H))
        self.C_bar = sp.sparse.csr_matrix(C_bar)

        self.d_bar = np.concatenate((du, dx)).repeat(H)

    def control(self, x0, z_ref, u_ref) -> Tuple[np.ndarray, dict]:
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
