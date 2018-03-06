import random

import numpy as np

from config import C_PUCT, ALPHA, EPSILON
from intefraces.i_game_state import IGameState
from intefraces.i_neural_net import INeuralNet
from utils.loaders import Game
from utils.loggers import mcts as log
from utils.progress import progress_bar


class Node:

    def __init__(self, state: IGameState):
        self.state = state
        self.edges = []


class Edge:

    def __init__(self, in_node: Node, out_node: Node, action: int, prior: float):
        self.in_node = in_node
        self.out_node = out_node
        self.action = action
        self.player = in_node.state.get_player()
        self.stats = {'N': 0, 'W': 0, 'Q': 0, 'P': prior}


class Mcts:
    """ Probabilistic Monte Carlo tree search """

    def __init__(self, nnet: INeuralNet):
        self._nnet = nnet
        self._tree = {}
        self._root = None

    def get_distribution(self, state: IGameState, simulations: int, verbose=False) -> np.ndarray:
        """ Perform MCTS simulations starting from current game state.
        :param state: root game state
        :param simulations: number of simulations
        :param verbose: print progress info
        :return visit_counts: a vector of probability distribution over all actions
        """
        # Set root node
        if state.get_hashable() not in self._tree:
            self._create_tree(state)
        else:
            self._root = self._tree[state.get_hashable()]
            self._prune_tree(self._root)
        # Explore the tree
        for i in range(simulations):
            log.info('********** Simulation {} **********'.format(i + 1))
            self._simulate()
            if verbose:
                progress_bar(i + 1, simulations, 'Exploring tree')
        # Return visit counts
        visit_counts = np.zeros(Game.ACTION_SIZE, dtype=np.integer)
        for edge in self._root.edges:
            visit_counts[edge.action] = edge.stats['N']
        log.info('---------- Search results ----------')
        for i in range(visit_counts.size):
            if visit_counts[i] > 0:
                log.info('Action {:4} | {:<4} Visit count'.format(i, visit_counts[i]))
        return visit_counts / sum(visit_counts)

    def _simulate(self):
        """
        This function performs one iteration of MCTS. It is descending the tree
        until a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, it is expanded, and the neural network is
        called to return an initial policy P and a value v for the node state.
        This value is propagated up the search path. In case the leaf node is
        a terminal state, the outcome is propagated up the search path. Edge
        stats of N, W, Q are updated.
        """
        log.info('Current player: {}'.format(self._root.state.get_player()))
        log.info('Root node')
        self._root.state.log(log)
        log.info('Moving to leaf...')
        leaf, breadcrumbs = self._move_to_leaf()
        if not leaf.state.is_game_finished():
            pi, value = self._nnet.predict(leaf.state.get_canonical_board(), leaf.state.get_valid_actions())
            log.info('Predicted value for player {}: {:.6f}'.format(leaf.state.get_player(), value))
            log.info('Expanding leaf...')
            self._expand_node(leaf, pi)
        else:
            value = leaf.state.get_value()
            log.info('True value for player {}: {}'.format(leaf.state.get_player(), value))
        log.info('Back propagating value...')
        self._back_propagate(leaf, value, breadcrumbs)

    def _move_to_leaf(self) -> (Node, list):
        """ Move down the tree until hit a leaf node """
        node = self._root
        breadcrumbs = []
        while node.edges:
            best_edge = self._get_best_edge(node)
            log.info('Player {} turn | Chosen action: {}'.format(node.state.get_player(), best_edge.action))
            node = best_edge.out_node
            breadcrumbs.append(best_edge)
        node.state.log(log)
        if node.state.is_game_finished():
            log.info('Game is finished')
        return node, breadcrumbs

    def _expand_node(self, leaf: Node, pi: np.ndarray):
        """ Expand node """
        for action in leaf.state.get_valid_actions():
            new_state = leaf.state.get_next_state(action)
            new_state.log(log)
            log.info('---------- Action {}'.format(action))
            # Add node if necessary
            if new_state.get_hashable() not in self._tree:
                new_node = Node(new_state)
                self._add_node(self._tree, new_node)
            else:
                new_node = self._tree[new_state.get_hashable()]
            # Add edge
            new_edge = Edge(leaf, new_node, action, pi[action])
            leaf.edges.append(new_edge)

    @staticmethod
    def _back_propagate(leaf: Node, value: float, breadcrumbs: list):
        """ Back propagate leaf value up the tree """
        leaf_player = leaf.state.get_player()
        for edge in breadcrumbs:
            sign = 1 if edge.player == leaf_player else -1
            old_value = edge.stats['W']
            edge.stats['N'] += 1
            edge.stats['W'] += value * sign
            edge.stats['Q'] = edge.stats['W'] / edge.stats['N']
            log.info('Updating edge with value {:9.6f} for player {}: N = {} | W = {:9.6f} | Q = {:9.6f}'.format(
                old_value, edge.player, edge.stats['N'], edge.stats['W'], edge.stats['Q']))

    def _get_best_edge(self, node: Node) -> Edge:
        """ Pick edge with highest upper confidence bound """
        total_visits = sum(edge.stats['N'] for edge in node.edges)
        max_u = -float('inf')
        candidates = []
        best_edge = None
        # Probability noise
        if node == self._root:
            nu = np.random.dirichlet([ALPHA] * len(node.edges))
            epsilon = EPSILON
        else:
            nu = [0] * len(node.edges)
            epsilon = 0
        for i, edge in enumerate(node.edges):
            p = edge.stats['P'] * (1 - epsilon) + nu[i] * epsilon
            u = edge.stats['Q'] + C_PUCT * p * total_visits ** 0.5 / (1 + edge.stats['N'])
            log.info('Action {:2}: N = {:2} | P = {:7.4f} | W = {:7.4f} | Q = {:7.4f} | U = {:7.4f}'.format(
                edge.action, edge.stats['N'], p, edge.stats['W'], edge.stats['Q'], u))
            if u > max_u:
                max_u = u
                candidates = []
                best_edge = edge
            elif u == max_u:
                candidates.append(edge)
        if candidates:
            return random.choice(candidates)  # tie break
        return best_edge

    def _create_tree(self, state: IGameState):
        """ Create a new tree """
        self._tree = {}
        self._root = Node(state)
        self._add_node(self._tree, self._root)

    def _prune_tree(self, node: Node):
        """ Keep only subtree of the node and prune the rest """

        def copy_subtree(node_: Node):
            for edge in node_.edges:
                self._add_node(subtree, edge.out_node)
                copy_subtree(edge.out_node)

        subtree = {}
        self._add_node(subtree, node)
        copy_subtree(node)
        self._tree = subtree

    @staticmethod
    def _add_node(tree: dict, node: Node):
        """ Add node to the tree """
        tree[node.state.get_hashable()] = node
