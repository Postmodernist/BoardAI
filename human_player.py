from player import Player


class HumanPlayer(Player):
    """ Player controlled by human """

    def __init__(self, name, board_size):
        super().__init__(name, board_size)

    def make_move(self, state, stochastic):
        """ Request the player to input action """
        action = -1
        while action not in state.allowed_actions:
            action = int(input('{}: '.format(state.allowed_actions)))
        return action, None, None, None
