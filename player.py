class Player:
    """ Player interface """

    def __init__(self, name, board_size):
        self.name = name
        self.board_size = board_size
        self.nn = None
        self.mcts = None

    def make_move(self, state, stochastic):
        """ Return intended move data
        :return action              Action data
        :return actions_prob_dist   Probability distribution over allowed actions
        :return mcts_value          Action value returned by MCTS
        :return nn_value            Action value predicted by neural net
        """
        return -1, None, None, None
