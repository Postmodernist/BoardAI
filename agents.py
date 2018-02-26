from game import State
from mcts import Mcts
from model import ResidualCnn
from player import Player


class Hel(Player):
    """ AI agent with intuition """

    def __init__(self, name: str, nn: ResidualCnn):
        super().__init__(name)
        self.nn = nn
        self.mcts = None

    def make_move(self, state: State, stochastic=True):
        """ Run MCTS to choose action """
        if self.mcts is None:
            self.mcts = Mcts(self.nn, state)
        return self.mcts.get_action(state, stochastic)


class Human(Player):
    """ Player controlled by human """

    def __init__(self, name: str):
        super().__init__(name)

    def make_move(self, state: State, _):
        """ Request the player to input action """
        action = -1
        while action not in state.allowed_actions:
            action = int(input('{}: '.format(state.allowed_actions)))
        return action, None, None, None
