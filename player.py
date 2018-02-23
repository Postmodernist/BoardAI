class Player:
    """ Player interface """

    def __init__(self, name):
        self.name = name
        self.nn = None
        self.mcts = None

    def make_move(self, state, stochastic):
        """ Return intended move data
        :param state                Current game state
        :param stochastic           Choose action stochastically or deterministically
        :return action              Chosen action
        :return actions_prob_dist   Probability distribution over actions
        :return mcts_value          Action value returned by MCTS
        :return nn_value            Action value predicted by neural net
        :rtype action: int
        :rtype actions_prob_dist: np.ndarray
        :rtype mcts_value: float
        :rtype nn_value: float
        """
        return -1, None, None, None
