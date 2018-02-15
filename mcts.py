import numpy as np
import config
from loggers import log_mcts


class Node:

    def __init__(self, state):
        self.state = state
        self.edges = []

    def is_leaf(self):
        return len(self.edges) == 0


class Edge:

    def __init__(self, in_node, out_node, prior, action):
        self.id = in_node.state.id + '|' + out_node.state.id
        self.in_node = in_node
        self.out_node = out_node
        self.player = in_node.state.player
        self.action = action
        self.stats = {'N': 0, 'W': 0, 'Q': 0, 'P': prior}


class Mcts:

    def __init__(self, root, cpuct):
        self.root = root
        self.tree = {}
        self.cpuct = cpuct
        self.add_node(root)

    def __len__(self):
        return len(self.tree)

    def add_node(self, node):
        self.tree[node.state.id] = node

    def move_to_leaf(self):
        log_mcts.info('Moving to leaf')
        breadcrumbs = []
        current_node = self.root
        finished = False
        value = 0
        while not current_node.is_leaf():
            log_mcts.info('Player %d turn...', current_node.state.player)
            max_qu = -99999
            if current_node == self.root:
                epsilon = config.EPSILON
                nu = np.random.dirichlet([config.ALPHA] * len(current_node.edges))  # exploration noise
            else:
                epsilon = 0
                nu = [0] * len(current_node.edges)
            nn = 0
            for action, edge in current_node.edges:
                nn += edge.stats['N']
            for idx, (action, edge) in enumerate(current_node.edges):
                u = self.cpuct * ((1 - epsilon) * edge.stats['P'] + epsilon * nu[idx]) * np.sqrt(nn) / (
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
            # The value of the newState from the POV of the new playerTurn
            new_state, value, finished = current_node.state.make_turn(simulation_action)
            current_node = simulation_edge.out_node
            breadcrumbs.append(simulation_edge)
        log_mcts('Finished... %d', finished)
        return current_node, value, finished, breadcrumbs

    @staticmethod
    def back_propagate(leaf, value, breadcrumbs):
        log_mcts.info('Back propagating...')
        current_player = leaf.state.player
        for edge in breadcrumbs:
            player = edge.player
            if player == current_player:
                sign = 1
            else:
                sign = -1
            edge.stats['N'] += 1
            edge.stats['W'] += value * sign
            edge.stats['Q'] = edge.stats['W'] / edge.stats['N']
            log_mcts.info('Updating edge with value %f for player %d... N = %d, W = %f, Q = %f',
                          value * sign, player, edge.stats['N'], edge.stats['W'], edge.stats['Q'])
            edge.out_node.state.render(log_mcts)
