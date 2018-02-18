import numpy as np

from player import Player


class HumanPlayer(Player):
    """ Player controlled by human """

    def __init__(self, name, action_size):
        super().__init__(name, action_size)

    def make_move(self, state, stochastic):
        """ Request the player to input action """
        print(state)
        action = -1
        while action not in state.allowed_actions:
            action = int(input(self.name + ': '))
        return action, None, None, None
