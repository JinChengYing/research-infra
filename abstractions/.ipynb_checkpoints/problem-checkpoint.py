# abstractions/problem.py

from abc import ABC, abstractmethod


class Problem(ABC):
    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def sample(self, n: int):
        pass
