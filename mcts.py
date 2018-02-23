import numpy as np

import config
import log


class Node:
    """ MCTS tree node """

    def __init__(self, state):
        self.state = state
        self.edges = []  # (action, edge) tuples

    def add_edge(self, action, edge):
        self.edges.append((action, edge))


class Edge:
    """ MCTS tree edge """

    def __init__(self, in_node, out_node, action, prior):
        self.in_node = in_node
        self.out_node = out_node
        self.action = action
        self.id = in_node.state.id + '|' + out_node.state.id
        self.player = in_node.state.player
        self.stats = {'N': 0, 'W': 0, 'Q': 0, 'P': prior}


class Mcts:
    """ Monte Carlo tree search """

    def __init__(self, state, nn):
        self.root = Node(state)
        self.nn = nn
        self.tree = {}
        self._add_node(self.root)

    def __len__(self):
        return len(self.tree)

    def simulate(self):
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
            total_visits = sum(edge.stats['N'] for _, edge in node.edges)
            if total_visits == 0:
                # Choose by Q
                chosen_edge = max(node.edges, key=lambda x: x[1].stats['Q'])[1]
            else:
                # Choose by Q + U
                chosen_edge = self._get_edge_with_max_qu(node, total_visits)
            log.mcts.info('Action with highest Q + U: {}'.format(chosen_edge.action))
            node = chosen_edge.out_node
            root_to_leaf_edges.append(chosen_edge)
        log.mcts.info('Game is finished: {}'.format(node.state.finished))
        return node, root_to_leaf_edges

    def _get_edge_with_max_qu(self, node, total_visits):
        """ Return node edge with max Q + U score
            :param total_visits     total number of simulations made from the node
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
        for i, (action, edge) in enumerate(node.edges):
            q = edge.stats['Q']
            p = (1 - epsilon) * edge.stats['P'] + epsilon * nu[i]  # P with noise
            if config.ALTERNATIVE_UCT:
                u = config.C * p * np.sqrt(total_visits) / (1 + edge.stats['N'])
            else:
                u = config.C * p * np.sqrt(np.log(total_visits) / (1 + edge.stats['N']))
            qu_sum = q + u  # probabilistic upper confidence tree score
            log.mcts.info(
                'Action {:2}: N {:2}, P {:.4f}, nu {:.4f}, P\' {:.4f}, W {:.4f}, Q {:.4f}, U {:.4f}, Q + U {:.4f}'
                    .format(action, edge.stats['N'], edge.stats['P'], nu[i], p, edge.stats['W'], q, u, qu_sum))
            if qu_sum > max_qu_sum:
                max_qu_sum = qu_sum
                chosen_edge = edge
        return chosen_edge

    def _expand_node(self, leaf_node, actions_prob_dist):
        """ Expand node """
        for i, action in enumerate(leaf_node.state.allowed_actions):
            new_state = leaf_node.state.make_move(action)
            # Add node if necessary
            if new_state.id not in self.tree:
                new_node = Node(new_state)
                self._add_node(new_node)
                log.mcts.info('New node {}, P = {:.6f}'.format(new_node.state.id, actions_prob_dist[action]))
            else:
                new_node = self.tree[new_state.id]
                log.mcts.info('Old node {}'.format(new_node.state.id))
            # Add edge
            new_edge = Edge(leaf_node, new_node, action, actions_prob_dist[action])
            leaf_node.add_edge(action, new_edge)

    @staticmethod
    def _back_propagate(leaf, state_value, root_to_leaf_edges):
        """ Back propagate leaf state value up the tree """
        leaf_player = leaf.state.player
        for edge in root_to_leaf_edges:
            if edge.player == leaf_player:
                sign = 1
            else:
                sign = -1
            old_state_value = edge.stats['W']
            edge.stats['N'] += 1
            edge.stats['W'] += state_value * sign
            edge.stats['Q'] = edge.stats['W'] / edge.stats['N']
            log.mcts.info('Updating edge with value {} for player {}: N = {}, W = {}, Q = {}'.format(
                old_state_value, edge.player, edge.stats['N'], edge.stats['W'], edge.stats['Q']))

    def _add_node(self, node):
        """ Add node to the MCTS tree """
        self.tree[node.state.id] = node
