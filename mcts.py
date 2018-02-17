import numpy as np

import config
from loggers import log_mcts


class Node:
    """ MCTS tree node """

    def __init__(self, state):
        self.state = state
        self.edges = []  # (action, edge) tuples

    def add_edge(self, action, edge):
        self.edges.append((action, edge))


class Edge:
    """ MCTS tree edge """

    def __init__(self, in_node, out_node, prior, action):
        self.in_node = in_node
        self.out_node = out_node
        self.action = action
        self.id = in_node.state.id + '|' + out_node.state.id
        self.player = in_node.state.player
        self.stats = {'N': 0, 'W': 0, 'Q': 0, 'P': prior}


class Mcts:
    """ Monte Carlo tree search facility """

    def __init__(self, state, nn):
        self.root = Node(state)
        self.nn = nn
        self.tree = {}
        self._add_node(self.root)

    def __len__(self):
        return len(self.tree)

    def simulate(self):
        """ Move to leaf node, evaluate it, and back propagate the value """
        log_mcts.info('Root node: %s', self.root.state.id)
        log_mcts.info(self.root.state)
        log_mcts.info('Current player: %d', self.root.state.player)
        # Move to the leaf node
        leaf, leaf_state_value, finished, root_to_leaf_edges = self._move_to_leaf()
        log_mcts.info(leaf.state)
        # Evaluate the leaf node
        log_mcts.info('Evaluating leaf...')
        if not finished:
            leaf_state_value, actions_prob_dist = self.get_nn_predictions(leaf.state)
            log_mcts.info('Predicted value for player %d: %f', leaf.state.player, leaf_state_value)
            # Expand the leaf node
            self._expand_node(leaf, actions_prob_dist)
        else:
            log_mcts.info('Game value for player %d: %f', leaf.player, leaf_state_value)
        # Back propagate the leaf node state value through the tree
        self._back_propagate(leaf, leaf_state_value, root_to_leaf_edges)

    def get_nn_predictions(self, state):
        """ Get state value and opponent move probability distribution predictions from NN """
        predictions = self.nn.predict(state)
        state_value = predictions[0][0]
        logits = predictions[1][0]
        # Mask invalid actions and simulate softmax layer
        odds = np.exp(logits[state.allowed_actions])
        actions_prob_dist = odds / np.sum(odds)
        return state_value, actions_prob_dist

    def _move_to_leaf(self):
        log_mcts.info('Moving to leaf')
        node = self.root
        node_state_value = 0
        finished = False
        root_to_leaf_edges = []
        while node.edges:
            log_mcts.info('Player %d turn...', node.state.player)
            max_qu = -99999
            if node == self.root:
                epsilon = config.EPSILON
                nu = np.random.dirichlet([config.ALPHA] * len(node.edges))  # exploration noise
            else:
                epsilon = 0
                nu = [0] * len(node.edges)
            nn = 0
            for action, edge in node.edges:
                nn += edge.stats['N']
            for idx, (action, edge) in enumerate(node.edges):
                u = config.CPUCT * ((1 - epsilon) * edge.stats['P'] + epsilon * nu[idx]) * np.sqrt(nn) / (
                        1 + edge.stats['N'])
                q = edge.stats['Q']
                log_mcts.info('Action: %d ... N = %d, P = %f, nu = %f, adjP = %f, W = %f, Q = %f, U = %f, Q+U = %f',
                              action, edge.stats['N'], round(edge.stats['P'], 6), round(nu[idx], 6),
                              ((1 - epsilon) * edge.stats['P'] + epsilon * nu[idx]), round(edge.stats['W'], 6),
                              round(q, 6), round(u, 6), round(q + u, 6))
                if q + u > max_qu:
                    max_qu = q + u
                    simulation_action = action
                    simulation_edge = edge
            log_mcts.info('Action with highest Q + U... %d', simulation_action)
            # The value of the new state from the POV of the new player
            _, node_state_value, finished = node.state.make_move(simulation_action)
            node = simulation_edge.out_node
            root_to_leaf_edges.append(simulation_edge)
        log_mcts('Finished... %d', finished)
        return node, node_state_value, finished, root_to_leaf_edges

    def _expand_node(self, leaf_node, actions_prob_dist):
        """ Expand node """
        log_mcts.info('Expanding leaf...')
        for i, action in enumerate(leaf_node.state.allowed_actions):
            new_state, _, _ = leaf_node.state.make_move(action)
            # Add node if necessary
            if new_state.id not in self.tree:
                new_node = Node(new_state)
                self._add_node(new_node)
                log_mcts.info('Added node %s, p = %f', new_node.state.id, actions_prob_dist[i])
            else:
                new_node = self.tree[new_state.id]
                log_mcts.info('Existing node %s', new_node.state.id)
            # Add edge
            new_edge = Edge(leaf_node, new_node, actions_prob_dist[i], action)
            leaf_node.add_edge(action, new_edge)

    @staticmethod
    def _back_propagate(leaf, state_value, root_to_leaf_edges):
        log_mcts.info('Back propagating leaf state value...')
        current_player = leaf.state.player
        for edge in root_to_leaf_edges:
            player = edge.player
            if player == current_player:
                sign = 1
            else:
                sign = -1
            edge.stats['N'] += 1
            edge.stats['W'] += state_value * sign
            edge.stats['Q'] = edge.stats['W'] / edge.stats['N']
            log_mcts.info('Updating edge with value %f for player %d... N = %d, W = %f, Q = %f',
                          state_value * sign, player, edge.stats['N'], edge.stats['W'], edge.stats['Q'])
            log_mcts.info(edge.out_node.state)

    def _add_node(self, node):
        """ Add node to MCTS tree """
        self.tree[node.state.id] = node
