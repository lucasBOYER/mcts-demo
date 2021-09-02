from __future__ import annotations
import numpy as np
from gym.envs.classic_control import PendulumEnv

from .base import BaseEnv


class DiscretePendulum(PendulumEnv, BaseEnv):
    def __init__(
        self,
        g=10,
        cur_step=0,
        max_step=50,
        torque_grid: list = None,
        n_actions: int = 7,
    ):
        super().__init__(g=g)
        self.cur_step = cur_step
        self.max_step = max_step
        if torque_grid is not None:
            torque_grid = np.clip(torque_grid, -self.max_torque, self.max_torque)
            self.torque_grid = torque_grid
        else:
            self.torque_grid = np.linspace(-self.max_torque, self.max_torque, n_actions)
        self.value = None
        self.turn = 0  # just to fit with the 2-ply oriented MCTS algo
        super().reset()  # will also set self.state, with theta and theta dot

    def copy(self) -> DiscretePendulum:
        c = DiscretePendulum(
            g=self.g,
            cur_step=self.cur_step,
            max_step=self.max_step,
            torque_grid=self.torque_grid,
        )
        c.state = self.state.copy()
        c.np_random.set_state(self.np_random.get_state())
        c.value = self.value
        return c

    def reset(self):
        super().reset()
        self.cur_step = 0

    def is_end(self) -> bool:
        return self.cur_step >= self.max_step

    def get_value(self, *args, **kwargs) -> bool:
        return self.value

    def get_legal_moves(self) -> list:
        return self.torque_grid

    def step_inplace(self, move):
        if not isinstance(move, list):
            move = [move]
        obs, reward, done, info = super().step(move)
        self.cur_step += 1
        self.value = reward
        return obs, reward, done, info

    def step(self, move):
        if not isinstance(move, list):
            move = [move]
        next_state = self.copy()
        next_state.cur_step += 1
        _, reward, _, _ = next_state.step_inplace(move)
        next_state.value = reward
        return next_state
