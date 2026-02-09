# abstractions/method.py
from abc import ABC, abstractmethod

class Method(ABC):
    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def run(self, problem):
        pass
