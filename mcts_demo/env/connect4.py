from __future__ import annotations

import numpy as np
from scipy.signal import convolve2d

from .base import BaseEnv

HORIZONTAL_KERNEL = np.array([[1, 1, 1, 1]])
VERTICAL_KERNEL = np.transpose(HORIZONTAL_KERNEL)
DIAG1_KERNEL = np.eye(4, dtype=np.uint8)
DIAG2_KERNEL = np.fliplr(DIAG1_KERNEL)
DETECTION_KERNELS = [HORIZONTAL_KERNEL, VERTICAL_KERNEL, DIAG1_KERNEL, DIAG2_KERNEL]


class ConnectEnv(BaseEnv):
    def __init__(self, size_x=7, size_y=6, turn=0, board=None):
        self.size_x, self.size_y = (
            board.shape if board is not None else (size_x, size_y)
        )
        self.board = (
            board
            if board is not None
            else np.repeat("_", size_x * size_y).reshape((size_x, size_y))
        )
        self.players = {0: "X", 1: "O"}
        self.turn = turn

    def reset(self):
        self.board = np.repeat("_", self.size_x * self.size_y).reshape(
            (self.size_x, self.size_y)
        )
        self.turn = 0

    def step(self, move: int, player: int = None) -> ConnectEnv:
        legal = self.get_legal_moves()
        if move not in legal:
            raise ValueError("Illegal move, column not available.")
        n_board = self.board.copy()
        # row exist as it is a legal move
        row = np.where(self.board[:, move] == "_")[0].min()
        n_board[(row, move)] = self.players.get(player or self.turn)
        return ConnectEnv(board=n_board, turn=int(not self.turn))

    def has_won(self, player: int) -> bool:
        board = self.board == self.players.get(player)
        for k in DETECTION_KERNELS:
            if (convolve2d(board, k, mode="valid") == 4).any():
                return True
        return False

    def is_end(self) -> bool:
        return len(self.get_legal_moves()) == 0 or self.has_won(0) or self.has_won(1)

    def get_value(self, pov: int) -> int:
        if self.has_won(pov):
            return 1  # win
        elif self.has_won(int(not pov)):
            return -1  # lose
        else:
            return 0  # draw

    def get_legal_moves(self) -> list:
        return [col for col, el in enumerate(self.board[-1, :]) if el == "_"]

    def render(self) -> None:
        print("   ".join(map(str, range(0, self.size_y))))
        for row in np.flip(self.board, axis=0):
            print(" | ".join(row))
