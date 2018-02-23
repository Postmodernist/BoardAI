from game import State
from player import Player


class HumanPlayer(Player):
    """ Player controlled by human """

    def __init__(self, name: str):
        super().__init__(name)

    def make_move(self, state: State, _):
        """ Request the player to input action """
        action = -1
        while action not in state.allowed_actions:
            action = int(input('{}: '.format(state.allowed_actions)))
        return action, None, None, None
