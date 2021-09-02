from mcts_demo.env.pendulum import DiscretePendulum
import numpy as np


def test_init():
    p = DiscretePendulum(g=9.8, cur_step=1, max_step=60, n_actions=5)
    max_torque = p.max_torque

    assert p.g == 9.8
    assert p.cur_step == 1
    assert p.max_step == 60
    assert len(p.torque_grid == 5)
    assert np.array_equal(p.torque_grid, np.linspace(-max_torque, max_torque, 5))

    # with a given torque grid that will be clipped
    grid = [-4, -3, -1, 0, 1, 3, 4]
    p = DiscretePendulum(torque_grid=grid)
    assert np.array_equal(p.torque_grid, [-2, -2, -1, 0, 1, 2, 2])

    assert p.state is not None
    assert np.array_equal(p.get_legal_moves(), [-2, -2, -1, 0, 1, 2, 2])


def test_copy():
    p = DiscretePendulum()
    copy = p.copy()

    assert np.array_equal(p.state, copy.state)
    p_rs = p.np_random.get_state()
    copy_rs = copy.np_random.get_state()

    for i in range(len(p_rs)):
        if not isinstance(p_rs[i], np.ndarray):
            assert p_rs[i] == copy_rs[i]
        else:
            assert np.array_equal(p_rs[i], copy_rs[i])

    copy.step_inplace(1)
    assert not np.array_equal(p.state, copy.state)


def test_step():
    p = DiscretePendulum()
    assert not p.is_end()
    for _ in range(p.max_step - 1):
        p.step_inplace(1)
        assert not p.is_end()
        assert not p.is_end()

    p.step_inplace(1)
    assert p.is_end()

    p.reset()
    assert p.cur_step == 0

    next_p = p.step(1)
    assert next_p.cur_step == 1
    assert not np.array_equal(p.state, next_p.state)
    assert p.value != next_p.value
    assert next_p.get_value() == next_p.value
