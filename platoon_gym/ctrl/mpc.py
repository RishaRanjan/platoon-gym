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
        Cf: Optional[np.ndarray] = None,
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
        self.n, self.m, self.p = n, m, p
        self.A, self.C = A, C
        self.Q, self.R, self.Qf, self.Cf = Q, R, Qf, Cf
        self.H = H
        self.opt_var_dim = H * (n + m)

        P = block_diag(*([R, C.T @ Q @ C] * (H - 1) + [R, C.T @ Qf @ C]))
        self.P = sp.sparse.csr_matrix(P)
        assert self.P.shape == (H * (p + m), H * (p + m))
        self.q = np.empty(self.opt_var_dim)

        A_bar = np.zeros((H * n, self.opt_var_dim))
        A_bar[:, :-n] += block_diag(*([-B] + [np.block([-A, -B])] * (H - 1)))
        A_bar[:-n, m:-n] += block_diag(
            *([np.block([np.eye(n), np.zeros((n, m))])] * (H - 1))
        )
        A_bar[-n:, -n:] += np.eye(n)
        if Cf is not None:
            A_bar = np.block([[A_bar], [np.zeros((n, self.opt_var_dim - n)), Cf]])
            assert A_bar.shape == ((H + 1) * n, self.opt_var_dim)
        else:
            assert A_bar.shape == (H * n, self.opt_var_dim)
        self.A_bar = sp.sparse.csr_matrix(A_bar)
        self.b_bar = np.zeros(self.A_bar.shape[0])

        C_bar = block_diag(*([Cu, Cx] * H))
        self.C_bar = sp.sparse.csr_matrix(C_bar)
        self.d_bar = np.tile(np.concatenate((du, dx)), H)
        assert self.C_bar.shape[0] == self.d_bar.shape[0]

    def control(
        self,
        x0: np.ndarray,
        z_ref: Optional[np.ndarray] = None,
        u_ref: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Returns a control input based on an MPC policy. Also returns a dict 
        with the planned trajectory and planned inputs.

        Args:
            x0: shape (n,), current state
            z_ref: shape (p, H+1), optional reference trajectory
            u_ref: shape (m, H), optional reference control input

        Returns:
            np.ndarray, shape (m,): control input
            dict: extra information related to the problem
        """
        n, m, p, H = self.n, self.m, self.p, self.H
        assert x0.ndim == 1 and x0.shape[0] == n
        if z_ref is not None:
            assert z_ref.ndim == 2 and z_ref.shape == (p, self.H + 1)
        else:
            z_ref = np.zeros((p, self.H + 1))
        if u_ref is not None:
            assert u_ref.ndim == 2 and u_ref.shape == (m, self.H)
        else:
            u_ref = np.zeros((m, self.H))

        prob, y = self.mpc_problem(x0, z_ref, u_ref)
        prob.solve(solver=cp.OSQP, warm_start=True)
        u_opt = np.empty((m, H))
        x_opt = np.empty((n, H + 1))
        if prob.status == cp.OPTIMAL:
            u_opt[:, 0] = y.value[:m]
            x_opt[:, 0] = x0
            for k, i in enumerate(range(m, m + (H - 1) * (m + n), m + n)):
                x_opt[:, k + 1] = y.value[i : i + n]
                u_opt[:, k + 1] = y.value[i + n : i + n + m]
            x_opt[:, -1] = y.value[-n:]
        info = {"status": prob.status, "planned states": x_opt, "planned inputs": u_opt}
        return np.atleast_1d(u_opt[:, 0]), info

    def mpc_problem(self, x0, z_ref, u_ref) -> Tuple[cp.Problem, cp.Variable]:
        """
        Creates the CVXPY problem and optimization variable for the MPC problem.
        """
        n, m, p, H = self.n, self.m, self.p, self.H
        y = cp.Variable(self.opt_var_dim)

        self.b_bar[: self.n] = self.A @ x0
        if self.Cf is not None:
            self.b_bar[-self.n :] = z_ref[:, -1]

        self.q[:m] = self.R @ u_ref[:, 0]
        for k, i in enumerate(range(m, m + (H - 1) * (m + n), m + n)):
            self.q[i : i + p] = self.C.T @ self.Q @ z_ref[:, k + 1]
            self.q[i + p : i + p + m] = self.R @ u_ref[:, k + 1]
        self.q[-p:] = self.C.T @ self.Qf @ z_ref[:, -1]
        self.q *= -2

        cost = cp.quad_form(y, self.P) + self.q @ y
        constraints = [self.A_bar @ y == self.b_bar, self.C_bar @ y <= self.d_bar]
        return cp.Problem(cp.Minimize(cost), constraints), y
