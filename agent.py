import numpy as np

import config
from loggers import log_mcts
from mcts import Mcts
from player import Player


class Agent(Player):
    """ Player controlled by AI """

    def __init__(self, name, action_size, nn):
        super().__init__(name, action_size)
        self.nn = nn
        self.mcts = None

    def make_move(self, state, stochastic=True):
        """ Run MCTS to evaluate potential actions, then choose action either stochastically or deterministically """
        # Setup MCTS tree
        if self.mcts is None or state.id not in self.mcts.tree:
            log_mcts.info('Building new MCTS tree for agent %s', self.name)
            self.mcts = Mcts(state, self.nn)
        else:
            log_mcts.info('Setting root of MCTS tree to %s for agent %s', state.id, self.name)
            self.mcts.root = self.mcts.tree[state.id]
        # Run the simulations
        for i in range(config.MCTS_SIMULATIONS):
            log_mcts.info('::: Simulation %d :::', i + 1)
            self.mcts.simulate()
        # Get actions values and probability distribution
        actions_values, actions_prob_dist = self._get_action_values()
        # Choose the action
        action, mcts_action_value = Agent._choose_action(actions_values, actions_prob_dist, stochastic)
        # Make a move
        next_state, _, _ = state.make_move(action)
        nn_action_value = -self.mcts.get_nn_predictions(next_state)[0]
        log_mcts.info('Actions probabilities: %s', actions_prob_dist)
        log_mcts.info('Chosen action: %d', action)
        log_mcts.info('MCTS perceived value: %f', mcts_action_value)
        log_mcts.info('NN perceived value: %f', nn_action_value)
        return action, actions_prob_dist, mcts_action_value, nn_action_value

    def _get_action_values(self):
        """ Values of root actions """
        edges = self.mcts.root.edges
        visit_counts = np.zeros(self.action_size, dtype=np.integer)
        actions_values = np.zeros(self.action_size, dtype=np.float32)
        for action, edge in edges:
            visit_counts[action] = edge.stats['N']
            actions_values[action] = edge.stats['Q']
        # Normalize visit counts, so that probabilities add up to 1
        actions_prob_dist = visit_counts / (np.sum(visit_counts) * 1.0)
        return actions_values, actions_prob_dist

    @staticmethod
    def _choose_action(actions_values, actions_prob_dist, stochastic=True):
        """ Choose action either stochastically or deterministically """
        if stochastic:
            # Choose stochastically using actions probabilities distribution
            actions_one_hot = np.random.multinomial(1, actions_prob_dist)
            action = np.where(actions_one_hot)[0][0]  # index of a non-zero element
        else:
            # Choose deterministically
            actions = np.argwhere(actions_prob_dist == max(actions_prob_dist))
            action = np.random.choice(actions)[0]  # tie break
        action_value = actions_values[action]
        return action, action_value
