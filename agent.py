import random

import numpy as np

import config
import log
from log_utils import print_actions_prob_dist
from mcts import Mcts
from player import Player


class Agent(Player):
    """ Player controlled by AI """

    def __init__(self, name, board_size, nn):
        super().__init__(name, board_size)
        self.nn = nn
        self.mcts = None

    def make_move(self, state, stochastic=True):
        """ Run MCTS to evaluate potential actions, then choose most promising action
        :type state                 game.State
        :param stochastic           choose action stochastically or deterministically
        :rtype action               int
        :rtype actions_prob_dist    1D array of size board_size
        :rtype mcts_action_value    float
        :rtype nn_action_value      float
        """

        # Setup MCTS tree
        if self.mcts is None or state.id not in self.mcts.tree:
            log.mcts.info('Building new MCTS tree for agent {}'.format(self.name))
            self.mcts = Mcts(state, self.nn)
        else:
            log.mcts.info('Setting root of MCTS tree to {} for agent {}'.format(state.id, self.name))
            self.mcts.root = self.mcts.tree[state.id]
        # Run the simulations
        for i in range(config.MCTS_SIMULATIONS):
            log.mcts.info('')
            log.mcts.info('********** Simulation {} **********'.format(i + 1))
            log.mcts.info('')
            self.mcts.simulate()
        # Get actions values and probability distribution
        action_values, actions_prob_dist = self._get_action_values_and_prob_dist()
        # Choose the action
        action = Agent._choose_action(actions_prob_dist, stochastic)
        # Make a move
        next_state = state.make_move(action)
        # Write action stats to log
        mcts_action_value = action_values[action]
        nn_action_value = -self.nn.predict(next_state)[0]
        log.mcts.info('')
        log.mcts.info('----- MCTS search results -----')
        print_actions_prob_dist(log.mcts, actions_prob_dist)
        log.mcts.info('Chosen action: {}'.format(action))
        log.mcts.info('MCTS perceived value: {}'.format(mcts_action_value))
        log.mcts.info('NN perceived value: {}'.format(nn_action_value))
        log.mcts.info('')
        return action, actions_prob_dist, mcts_action_value, nn_action_value

    def _get_action_values_and_prob_dist(self):
        """ Values and probabilities of root actions
        :rtype action_values        1D array of size board_size
        :rtype actions_prob_dist    1D array of size board_size
        """
        edges = self.mcts.root.edges
        visit_counts = np.zeros(self.board_size, dtype=np.integer)
        action_values = np.zeros(self.board_size, dtype=np.float64)
        for action, edge in edges:
            visit_counts[action] = edge.stats['N']
            action_values[action] = edge.stats['Q']
        # Normalize visit counts, so that probabilities add up to 1
        actions_prob_dist = visit_counts / (np.sum(visit_counts) * 1.0)
        return action_values, actions_prob_dist

    @staticmethod
    def _choose_action(actions_prob_dist, stochastic=True):
        """ Choose action
        :type actions_prob_dist     1D array of size board_size
        :param stochastic           if True then choose stochastically, otherwise -- deterministically
        :rtype action               int
        :rtype action_value         float
        """
        if stochastic:
            # Choose stochastically using actions probability distribution
            action_one_hot = np.random.multinomial(1, actions_prob_dist)
            action = np.where(action_one_hot)[0][0]  # index of a non-zero element
        else:
            # Choose deterministically
            actions = np.argwhere(actions_prob_dist == max(actions_prob_dist)).T[0]
            action = random.choice(actions)  # tie break
        return action
