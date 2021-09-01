import numpy as np
from typing import Tuple, List


class Vertex:
    def __init__(
        self,
        state,
        w: int = 0,
        n: int = 0,
        random_state: np.random.RandomState = None,
        prev_action: int = None,
    ):
        self.state = state
        self.w = w
        self.n = n
        self.children = []
        self.random_state = random_state or np.random.RandomState()
        self.prev_action = prev_action

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_terminal(self) -> bool:
        return self.state.is_end()

    def expand(self) -> None:
        if len(self.children) > 0:
            raise ValueError(
                f"Vertex already expanded, with {len(self.children)} children."
            )
        for m in self.state.get_legal_moves():
            self.children.append(
                Vertex(
                    state=self.state.step(m),
                    w=0,
                    n=0,
                    prev_action=m,
                    random_state=self.random_state,
                )
            )

    def rollout(self, pov: int) -> float:
        cur_state = self.state
        while not cur_state.is_end():
            moves = cur_state.get_legal_moves()
            cur_state = cur_state.step(self.random_state.choice(moves))
        return cur_state.get_value(pov=pov)

    def display(self, render: bool = True) -> None:
        if render:
            self.state.render()
        print(f"self.w={self.w}")
        print(f"self.n={self.n}")
        print(f"# of children : {len(self.children)}")


class MCTS:
    def __init__(
        self,
        root: Vertex,
        c: float = 2,
        random_state: np.random.RandomState = None,
        two_players: bool = True,
    ):
        self.root = root
        self.c = c
        self.random_state = random_state or np.random.RandomState()
        self.two_players = two_players

    def compute_uct(self, child, parent_N) -> float:
        if child.n == 0:
            return np.inf
        return child.w / child.n + self.c * np.sqrt(np.log(parent_N) / child.n)

    def selection(self) -> Tuple[Vertex, List[Vertex]]:
        cur_node = self.root
        path = [self.root]
        while not cur_node.is_leaf():
            cur_node = self.select_child(cur_node)
            path.append(cur_node)
        return cur_node, path

    def backpropagate(self, path, reward) -> None:
        # I prefer to evaluate the two_players property once and duplicate some code
        # rather than evaluate it len(path) times
        if self.two_players:
            for v in path:
                # The sign of the reward of a node must be flipped when the player
                # that took the action leading to this node is not the root player
                # i.e. when parent(node.state.turn) != root.state.turn
                # which is equivalent to  node.state.turn == root.state.turn
                sign = -1 if v.state.turn == self.root.state.turn else 1
                v.w += sign * reward
                v.n += 1
        else:
            for v in path:
                v.w += reward
                v.n += 1

    def select_child(self, parent) -> Vertex:
        children = parent.children
        ucts = [self.compute_uct(child, parent.n) for child in children]
        return children[np.argmax(ucts)]

    def search(self, n=50):
        for _ in range(n):
            # selection
            leaf, path = self.selection()
            if not leaf.is_terminal():
                # expansion
                leaf.expand()
                child = leaf.children[
                    self.random_state.choice(range(len(leaf.children)))
                ]
                path.append(child)
                # rollout
                reward = child.rollout(pov=self.root.state.turn)
            else:
                reward = leaf.state.get_value(pov=self.root.state.turn)
            # backpropagation
            self.backpropagate(path, reward)
        pi = [v.n for v in self.root.children]  # policy after search ~N_visits
        return self.root.children[np.argmax(pi)].prev_action
