from game import State
from mcts import Mcts
from model import ResidualCnn
from player import Player


class Agent(Player):
    """ Player controlled by AI """

    def __init__(self, name: str, nn: ResidualCnn):
        super().__init__(name)
        self.nn = nn
        self.mcts = None

    def make_move(self, state: State, stochastic=True):
        """ Run MCTS to choose most promising action
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
        # Create MCTS tree
        if self.mcts is None:
            self.mcts = Mcts(self.nn, state)
        # Choose action
        action, actions_prob_dist, mcts_value, nn_value = self.mcts.choose_action(state, stochastic)
        return action, actions_prob_dist, mcts_value, nn_value
