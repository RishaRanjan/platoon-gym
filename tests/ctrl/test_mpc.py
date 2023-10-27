import numpy as np

from platoon_gym.ctrl.mpc import LinearMPC


def test_mpc():
    A = np.array([[1, 1], [0, 1]])
    B = np.array([[0], [1]])
    C = np.eye(2)
    Q = np.eye(2)
    R = np.eye(1)
    Qf = np.eye(2)
    Cf = np.eye(2)
    Cx = np.block([[np.eye(2)], [-np.eye(2)]])
    Cu = np.block([[np.eye(1)], [-np.eye(1)]])
    dx = np.array([np.inf, np.inf, np.inf, np.inf])
    du = np.array([np.inf, np.inf])
    H = 3
    mpc = LinearMPC(A, B, C, Q, R, Qf, Cx, Cu, dx, du, H, Cf=Cf)
    u_opt, _ = mpc.control(np.array([0, 0]))
    assert np.allclose(u_opt, np.array([0]))
    mpc = LinearMPC(A, B, C, Q, R, Qf, Cx, Cu, dx, du, H)
    u_opt, _ = mpc.control(np.array([0, 0]))
    assert np.allclose(u_opt, np.array([0]))
