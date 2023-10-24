from abc import ABC, abstractmethod
import numpy as np


class ControllerBase(ABC):
    """
    Base vehicle controller class.
    """
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def control(self):
        pass