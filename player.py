class Player:
    """ Player interface """

    def __init__(self, name, board_size):
        self.name = name
        self.board_size = board_size

    def make_move(self, state, stochastic):
        """ Return intended move data """
        action = -1  # action data
        actions_prob_dist = None  # probability distribution over the allowed actions
        value = None  # action value for this player
        nn_value = None  # action value predicted by neural net
        return action, actions_prob_dist, value, nn_value
