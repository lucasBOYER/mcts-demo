from mcts_demo.env.connect4 import ConnectEnv
import numpy as np
from pytest import raises


def test_init():
    env = ConnectEnv()
    assert env.board.shape == (7, 6)
    assert (env.board == "_").all()


def test_terminal():
    empty_b = np.repeat("_", 7 * 6).reshape((7, 6))
    board_d1 = empty_b.copy()
    board_d2 = empty_b.copy()
    board_h = empty_b.copy()
    board_v = empty_b.copy()
    for i in range(4):
        board_d1[i, i] = "X"
        board_d2[6 - i, i] = "X"
        board_h[0, i] = "X"
        board_v[i, 0] = "X"

    for b in [board_d1, board_d2, board_h, board_v]:
        e = ConnectEnv(board=b)
        assert e.has_won(0)
        assert e.get_value(pov=0) == 1
        assert e.get_value(pov=1) == -1
        assert e.is_end()

    board_full = np.repeat("X", 7 * 6).reshape((7, 6))
    assert ConnectEnv(board=board_full).is_end()
    assert not ConnectEnv(board=empty_b).is_end()


def test_legal_moves():
    board = np.repeat("X", 7 * 6).reshape((7, 6))
    board[-1, :2] = "_"
    assert np.array_equal(ConnectEnv(board=board).get_legal_moves(), [0, 1])


def test_step():
    env = ConnectEnv()
    env = env.step(0)
    assert env.board[0, 0] == "X"
    assert env.get_value(pov=0) == 0

    env.reset()
    assert (env.board == "_").all()

    env.board[:, 0] = "X"
    with raises(ValueError):
        env.step(0)
