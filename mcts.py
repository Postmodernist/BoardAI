import random

import numpy as np

import config
import log
from game import Game, State
from model import ResidualCnn
from log_utils import print_actions_prob_dist


class Node:
    """ MCTS tree node """

    def __init__(self, state: State):
        self.state = state
        self.edges = []


class Edge:
    """ MCTS tree edge """

    def __init__(self, in_node: Node, out_node: Node, action: int, prior: float):
        self.in_node = in_node
        self.out_node = out_node
        self.action = action
        self.player = in_node.state.player
        self.stats = {'N': 0, 'W': 0, 'Q': 0, 'P': prior}


class Mcts:
    """ Monte Carlo tree search """

    def __init__(self, nn: ResidualCnn, state: State):
        self.nn = nn
        self.root = Node(state)
        self.tree = {}
        self._add_node(self.root)

    def __len__(self):
        return len(self.tree)

    def choose_action(self, state: State, stochastic=True):
        """ Search for the most promising action """
        # Set root node
        if state.id not in self.tree:
            self._reset_tree(state)
        else:
            self.root = self.tree[state.id]
        # Run the simulations
        for i in range(config.MCTS_SIMULATIONS):
            log.mcts.info('')
            log.mcts.info('********** Simulation {} **********'.format(i + 1))
            log.mcts.info('')
            self._simulate()
        # Get actions values and probability distribution
        action_values, actions_prob_dist = self._get_action_values_and_prob_dist()
        # Choose the action
        action = self._choose_action(actions_prob_dist, stochastic)
        # Make a move
        next_state = state.make_move(action)
        # Write action stats to log
        mcts_value = action_values[action]
        nn_value = -self.nn.predict(next_state)[0]
        log.mcts.info('')
        log.mcts.info('----- MCTS search results -----')
        print_actions_prob_dist(log.mcts, actions_prob_dist)
        log.mcts.info('Action: {} | MCTS val: {:.6f} | NN val: {:.6f}'.format(action, mcts_value, nn_value))
        log.mcts.info('')
        return action, actions_prob_dist, mcts_value, nn_value

    def _reset_tree(self, state: State):
        """ Reset MCTS tree """
        self.root = Node(state)
        self.tree = {}
        self._add_node(self.root)

    def _add_node(self, node: Node):
        """ Add node to the MCTS tree """
        self.tree[node.state.id] = node

    def _simulate(self):
        """ Move to leaf node, evaluate it, and back propagate the value """
        log.mcts.info('Root node: {}'.format(self.root.state.id))
        log.mcts.info('Current player: {}'.format(self.root.state.player))
        log.mcts.info('Moving to leaf...')
        leaf, root_to_leaf_edges = self._move_to_leaf()
        log.mcts.info('Evaluating leaf...')
        leaf_state_value = leaf.state.value
        if not leaf.state.finished:
            leaf_state_value, actions_prob_dist = self.nn.predict(leaf.state)
            log.mcts.info('Predicted value for player {}: {}'.format(leaf.state.player, leaf_state_value))
            log.mcts.info('Expanding leaf...')
            self._expand_node(leaf, actions_prob_dist)
        else:
            log.mcts.info('Final game value for player {}: {}'.format(leaf.state.player, leaf_state_value))
        log.mcts.info('Back propagating leaf state value...')
        self._back_propagate(leaf, leaf_state_value, root_to_leaf_edges)

    def _move_to_leaf(self):
        """ Move down the tree until hit the leaf node """
        node = self.root
        root_to_leaf_edges = []
        while node.edges:
            log.mcts.info('Player {} turn'.format(node.state.player))
            total_visits = sum(edge.stats['N'] for edge in node.edges)
            if total_visits == 0:
                # Choose by Q
                chosen_edge = max(node.edges, key=lambda x: x.stats['Q'])
            else:
                # Choose by Q + U
                chosen_edge = self._get_edge_with_max_qu(node, total_visits)
            log.mcts.info('Action with highest Q + U: {}'.format(chosen_edge.action))
            node = chosen_edge.out_node
            root_to_leaf_edges.append(chosen_edge)
        log.mcts.info('Game is finished: {}'.format(node.state.finished))
        return node, root_to_leaf_edges

    def _get_edge_with_max_qu(self, node: Node, total_visits: int):
        """ Return node edge with max Q + U score
            :param node             Node to examine
            :param total_visits     Total number of simulations made from the node
        """
        # epsilon: noise ratio, [0...1]
        # nu: noise values array of size len(node.edges)
        if node == self.root:
            epsilon = config.EPSILON
            nu = np.random.dirichlet([config.ALPHA] * len(node.edges))  # exploration noise
        else:
            epsilon = 0
            nu = [0] * len(node.edges)
        max_qu_sum = -99999
        chosen_edge = None
        # Choose best edge
        for i, edge in enumerate(node.edges):
            q = edge.stats['Q']
            p = (1 - epsilon) * edge.stats['P'] + epsilon * nu[i]  # P with noise
            u = config.C * p * np.sqrt(np.log(total_visits) / (1 + edge.stats['N']))
            qu_sum = q + u  # probabilistic upper confidence tree score
            log.mcts.info(
                'Action {:2}: N {:2}, P {:.4f}, nu {:.4f}, P\' {:.4f}, W {:.4f}, Q {:.4f}, U {:.4f}, Q + U {:.4f}'
                    .format(edge.action, edge.stats['N'], edge.stats['P'], nu[i], p, edge.stats['W'], q, u, qu_sum))
            if qu_sum > max_qu_sum:
                max_qu_sum = qu_sum
                chosen_edge = edge
        return chosen_edge

    def _expand_node(self, leaf: Node, actions_prob_dist: np.ndarray):
        """ Expand node """
        for action in leaf.state.allowed_actions:
            new_state = leaf.state.make_move(action)
            # Add node if necessary
            if new_state.id not in self.tree:
                new_node = Node(new_state)
                self._add_node(new_node)
                log.mcts.info('New node {}, P = {:.6f}'.format(new_node.state.id, actions_prob_dist[action]))
            else:
                new_node = self.tree[new_state.id]
                log.mcts.info('Old node {}'.format(new_node.state.id))
            # Add edge
            new_edge = Edge(leaf, new_node, action, actions_prob_dist[action])
            leaf.edges.append(new_edge)

    @staticmethod
    def _back_propagate(leaf: Node, state_value: float, root_to_leaf_edges: list):
        """ Back propagate leaf state value up the tree """
        leaf_player = leaf.state.player
        for edge in root_to_leaf_edges:
            sign = 1 if edge.player == leaf_player else -1
            old_state_value = edge.stats['W']
            edge.stats['N'] += 1
            edge.stats['W'] += state_value * sign
            edge.stats['Q'] = edge.stats['W'] / edge.stats['N']
            log.mcts.info('Updating edge with value {} for player {}: N = {}, W = {}, Q = {}'.format(
                old_state_value, edge.player, edge.stats['N'], edge.stats['W'], edge.stats['Q']))

    def _get_action_values_and_prob_dist(self):
        """ Values and probabilities of root actions
        :return action_values        1D array of size board_size
        :return actions_prob_dist    1D array of size board_size
        :rtype action_values: np.ndarray
        :rtype actions_prob_dist: np.ndarray
        """
        edges = self.root.edges
        visit_counts = np.zeros(Game.board_size, dtype=np.integer)
        action_values = np.zeros(Game.board_size, dtype=np.float64)
        for edge in edges:
            visit_counts[edge.action] = edge.stats['N']
            action_values[edge.action] = edge.stats['Q']
        # Normalize visit counts, so that probabilities add up to 1
        actions_prob_dist = visit_counts / np.sum(visit_counts)
        return action_values, actions_prob_dist

    @staticmethod
    def _choose_action(actions_prob_dist: np.ndarray, stochastic=True):
        """ Choose action
        :param actions_prob_dist    1D array of size board_size, Pi of actions
        :param stochastic           if True then choose stochastically, otherwise -- deterministically
        :rtype action               int
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
