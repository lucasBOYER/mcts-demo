import numpy as np
from pytest import raises

from mcts_demo.env.connect4 import ConnectEnv
from mcts_demo.mcts import MCTS, Vertex


def test_vertex():
    env = ConnectEnv()
    v = Vertex(env)
    assert v.is_leaf()
    assert not v.is_terminal()

    v.expand()
    assert not v.is_leaf()
    assert len(v.children) == env.size_y  # all cols free

    with raises(ValueError):  # already expanded
        v.expand()

    c = v.children[0]
    assert c.rollout(pov=0) in [-1, 1, 0]
    assert c.prev_action == 0


def test_mcts():
    env = ConnectEnv()
    root = Vertex(env)
    mcts = MCTS(root=root, two_players=True)

    cur_node, path = mcts.selection()
    assert cur_node == root
    assert path == [root]

    # Simulate a case when root and child 0 has been explored
    root.expand()
    root.children[0].n = 1
    root.children[0].w = 1
    root.n = 1  # and root.w does not matter

    # log(1) == 0 so only the w part of UCT formula will count for the explored child,
    # others will be np.inf
    # UCT = child.w / child.n + self.c * np.sqrt(np.log(parent_N) / child.n))
    assert mcts.compute_uct(root.children[0], 1) == 1
    assert all([mcts.compute_uct(root.children[i], 1) == np.inf for i in range(1, 6)])

    cur_node, path = mcts.selection()
    assert len(path) == 2
    assert cur_node == root.children[1]
    assert path == [root, root.children[1]]

    mcts.backpropagate(path, reward=1)
    assert root.children[1].n == 1
    assert root.children[1].w == 1
    assert root.n == 2
    assert root.w == -1  # flipped even though does not matter for the root


def test_search():
    env = ConnectEnv()
    root = Vertex(env)
    mcts = MCTS(root=root, two_players=True)

    _ = mcts.search(n=1)
    assert mcts.root.n == 1
    assert sum([c.n for c in root.children]) == 1


def test_not_dumb():
    env = ConnectEnv()
    env.board[:3, 0] = "O"
    rs = np.random.RandomState(0)
    mcts = MCTS(root=Vertex(env, random_state=rs), random_state=rs)
    assert mcts.search(n=100) == 0  # X loses if it doesn't play 0


def test_display():
    # just checking that it does not crash
    env = ConnectEnv()
    v = Vertex(env)
    v.display()
