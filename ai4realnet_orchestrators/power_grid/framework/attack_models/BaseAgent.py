from abc import ABC, abstractmethod
from typing import Union
from pathlib import Path


class BaseAgent(ABC):
    def __init__(self, name):
        super().__init__()
        self.name = name

    @abstractmethod
    def load(self, path: Union[str, Path]):
        pass

    def perturb(self, obs):
        pass

    def train(self, **kwargs):
        pass

    def evaluate(self, **kwargs):
        pass
