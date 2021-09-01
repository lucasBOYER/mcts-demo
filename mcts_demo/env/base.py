from __future__ import annotations
from abc import ABC, abstractmethod


class BaseEnv(ABC):  # pragma: no cover
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, move) -> BaseEnv:
        pass

    @abstractmethod
    def is_end(self) -> bool:
        pass

    @abstractmethod
    def get_value(self) -> float:
        pass

    @abstractmethod
    def get_legal_moves(self) -> list:
        pass

    @abstractmethod
    def render(self):
        pass
