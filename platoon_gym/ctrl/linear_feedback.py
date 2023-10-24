import numpy as np

from platoon_gym.ctrl.controller_base import ControllerBase


class LinearFeedback(ControllerBase):
    """
    Linear feedback controller. Uses sum of positive gains multiplied by error 
    signals to determine control action.

    :param k: shape (m, n), n is state (or error vector) size, m is input size
    """
    def __init__(self, k: np.ndarray):
        assert k.ndim == 2
        self.k = k
        self.m = self.k.shape[0]
    
    def control(self, e: np.ndarray) -> np.ndarray:
        """
        Returns a control input based on a linear feedback control policy.

        :param e: shape (n,), error or state (if driving state to zero) vector
        :return: shape (m,), control input
        """
        return -self.k @ e