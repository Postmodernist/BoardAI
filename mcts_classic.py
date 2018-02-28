import math
import random

import numpy as np

import config
from game import Game, State, BOARD_SIDE, INDEX_BOARD
from log_utils import progress_bar

SIMULATIONS = config.MCTS_CLASSIC_SIMULATIONS
VERBOSE = config.MCTS_CLASSIC_VERBOSE


def get_neighbors():
    """ Get neighbors for each position """
    # Create framed board
    fb_side = BOARD_SIDE + 2
    framed_board = np.full(fb_side ** 2, -1)
    framed_board.reshape(fb_side, fb_side)[1:-1, 1:-1] = INDEX_BOARD.reshape(BOARD_SIDE, BOARD_SIDE)
    # Get neighbors
    neighbors = {}
    for i in np.arange(fb_side ** 2).reshape(fb_side, fb_side)[1:-1, 1:-1].ravel():
        ni = [i - fb_side - 1, i - fb_side, i - fb_side + 1, i - 1, i + 1, i + fb_side - 1, i + fb_side,
              i + fb_side + 1]
        neighbors[framed_board[i]] = list(filter(lambda x: x != -1, framed_board[ni]))
    return neighbors


NEIGHBORS = get_neighbors()


class Node:
    """ MCTS tree node """

    def __init__(self, state: State):
        self.state = state
        self.edges = []


class Edge:
    """ MCTS tree edge """

    def __init__(self, in_node: Node, out_node: Node, action: int):
        self.in_node = in_node
        self.out_node = out_node
        self.action = action
        self.player = in_node.state.player
        self.stats = {'N': 0, 'W': 0, 'Q': 0}


class MctsClassic:
    """ Monte Carlo tree search """

    def __init__(self, state: State):
        self.root = Node(state)
        self.tree = {}
        self._add_node(self.tree, self.root)

    def __len__(self):
        return len(self.tree)

    def get_action(self, state: State):
        """ Search for the most promising action """
        # Set root node
        if state.id not in self.tree:
            self._reset_tree(state)
        else:
            self.root = self.tree[state.id]
            self._prune_tree(self.root)  # cleanup the tree
        # Explore the tree
        if VERBOSE:
            print('Exploring tree...')
        for i in range(SIMULATIONS):
            self._simulate()
            if VERBOSE:
                progress_bar(i + 1, SIMULATIONS)
        # Choose the action
        action, mcts_value, actions_prob_dist = self._choose_action()
        return action, actions_prob_dist, mcts_value, None

    @staticmethod
    def _add_node(tree: dict, node: Node):
        """ Add node to the tree """
        tree[node.state.id] = node

    def _reset_tree(self, state: State):
        """ Reset the tree """
        self.root = Node(state)
        self.tree = {}
        self._add_node(self.tree, self.root)

    def _prune_tree(self, node: Node):
        """ Keep only subtree of the node and prune the rest """

        def copy_subtree(node_: Node):
            for edge in node_.edges:
                self._add_node(subtree, edge.out_node)
                copy_subtree(edge.out_node)

        subtree = {}
        self._add_node(subtree, node)
        copy_subtree(node)
        self.tree = subtree

    def _simulate(self):
        """ Move to leaf node, evaluate it, and back propagate the value """
        leaf, root_to_leaf_edges = self._move_to_leaf()
        if leaf.state.finished:
            leaf_state_value = leaf.state.value
        else:
            leaf_state_value = self._rollout(leaf.state)
            self._expand_node(leaf)
        self._back_propagate(leaf, leaf_state_value, root_to_leaf_edges)

    def _move_to_leaf(self):
        """ Move down the tree until hit the leaf node """
        node = self.root
        root_to_leaf_edges = []
        while node.edges:
            chosen_edge = self._get_edge_with_max_qu(node)
            node = chosen_edge.out_node
            root_to_leaf_edges.append(chosen_edge)
        return node, root_to_leaf_edges

    @staticmethod
    def _get_edge_with_max_qu(node: Node):
        """ Return node edge with max Q + U score """
        total_visits = sum(edge.stats['N'] for edge in node.edges)
        max_qu_sum = -99999
        chosen_edge = None
        # Choose best edge
        for edge in node.edges:
            if edge.stats['N'] == 0:
                return edge
            q = edge.stats['Q']
            u = config.MCTS_C * (math.log(total_visits) / edge.stats['N']) ** 0.5
            qu_sum = q + u  # upper confidence tree score
            if qu_sum > max_qu_sum:
                max_qu_sum = qu_sum
                chosen_edge = edge
        return chosen_edge

    @staticmethod
    def _rollout(state: State):
        """ Return a result of a random rollout """
        board = state.board.copy()
        player = state.player
        valid_actions = set(state.allowed_actions)
        # Random descent until game end
        while not State.is_player_won(board, -player):
            # Make random move
            action = random.choice(list(valid_actions))
            board[action] = player
            player = -player
            # Update valid actions
            valid_actions.remove(action)
            empty_neighbors = [x for x in NEIGHBORS[action] if board[x] == 0]
            valid_actions.update(empty_neighbors)
            if len(valid_actions) == 0:
                # Game draw
                return 0
        return -1 if player == state.player else 1

    def _expand_node(self, leaf: Node):
        """ Expand node """
        for action in leaf.state.allowed_actions:
            new_state = leaf.state.make_move(action)
            # Add node if necessary
            if new_state.id not in self.tree:
                new_node = Node(new_state)
                self._add_node(self.tree, new_node)
            else:
                new_node = self.tree[new_state.id]
            # Add edge
            new_edge = Edge(leaf, new_node, action)
            leaf.edges.append(new_edge)

    @staticmethod
    def _back_propagate(leaf: Node, state_value: int, root_to_leaf_edges: list):
        """ Back propagate leaf state value up the tree """
        leaf_player = leaf.state.player
        for edge in root_to_leaf_edges:
            sign = 1 if edge.player == leaf_player else -1
            edge.stats['N'] += 1
            edge.stats['W'] += state_value * sign
            edge.stats['Q'] = edge.stats['W'] / edge.stats['N']

    def _choose_action(self):
        """ Choose action
        :rtype action: int
        :rtype action_value: float
        :rtype actions_prob_dist: np.ndarray
        """
        edges = self.root.edges
        visit_counts = np.zeros(Game.board_size, dtype=np.integer)
        action_values = np.zeros(Game.board_size, dtype=np.float64)
        for edge in edges:
            visit_counts[edge.action] = edge.stats['N']
            action_values[edge.action] = edge.stats['Q']
        # Choose deterministically by visit counts
        actions_prob_dist = visit_counts / sum(visit_counts)
        actions = np.argwhere(visit_counts == max(visit_counts)).T[0]
        action = random.choice(actions)  # tie break
        action_value = action_values[action]
        return action, action_value, actions_prob_dist
