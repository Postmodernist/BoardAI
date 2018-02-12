import numpy as np
import random
import mcts as mc
from game import State
from loss import softmax_cross_entropy_with_logits
import config
from loggers import log_mcts
import time
import matplotlib.pyplot as plt


class User:

    def __init__(self, name, state_size, action_size):
        self.name = name
        self.state_size = state_size
        self.action_size = action_size

    def act(self, state, tau):
        action = input('Enter your action: ')
        pi = np.zeros(self.action_size)
        pi[action] = 1
        value = None
        nn_value = None
        return action, pi, value, nn_value


class Agent:

    def __init__(self, name, state_size, action_size, mcts_simulations, cpuct, model):
        self.name = name
        self.state_size = state_size
        self.action_size = action_size
        self.mcts_simulations = mcts_simulations
        self.model = model
        self.mcts = None
        self.train_overall_loss = []
        self.train_value_loss = []
        self.train_policy_loss = []
        self.val_overall_loss = []
        self.val_value_loss = []
        self.val_policy_loss = []

    def simulate(self):
        log_mcts.info()
        log_mcts.info('Root node...%s', self.mcts.root.state.id)
        self.mcts.root.state.render(log_mcts)
        log_mcts.info('Current player...%d', self.mcts.root.state.playerTurn)
        # Move to the leaf node
        leaf, value, done, breadcrumbs = self.mcts.moveToLeaf()
        leaf.state.render(log_mcts)
        # Evaluate the leaf node
        value, breadcrumbs = self.evaluate_leaf(leaf, value, done, breadcrumbs)
        # Back propagate the value through the tree
        self.mcts.back_propagate(leaf, value, breadcrumbs)

# TODO: finish Agent
