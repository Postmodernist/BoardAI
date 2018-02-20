import random
from collections import deque

import log


class Memory:

    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.long_memory = deque(maxlen=memory_size)
        self.short_memory = deque(maxlen=memory_size)

    def commit_short_memory(self, identities):
        for identity in identities:
            state = identity[0]
            actions_values = identity[1]
            item = {
                'state': state,
                'actions_values': actions_values,
                'value': None}
            self.short_memory.append(item)

    def commit_long_memory(self):
        self.long_memory.extend(self.short_memory)
        self.short_memory.clear()

    def log_long_memory_sample(self, current_player, best_player):
        log.memory.info('----------------------------------------')
        log.memory.info('New memories')
        log.memory.info('----------------------------------------')
        log.memory.info('')
        memory_sample = random.sample(self.long_memory, min(1000, len(self.long_memory)))
        for mem in memory_sample:
            current_value, current_prob_dist = current_player.nn.predict(mem['state'])
            best_value, best_prob_dist = best_player.nn.predict(mem['state'])
            mem['state'].log(log.memory)
            log.memory.info('MCTS simulated value for player {}: {}'.format(mem['state'].player, mem['value']))
            log.memory.info('Cur. predicted value for player {}: {}'.format(mem['state'].player, current_value))
            log.memory.info('Best predicted value for player {}: {}'.format(mem['state'].player, best_value))
            log.memory.info(
                'MCTS simulated actions values: {}'.format(['{:.2f}'.format(x) for x in mem['actions_values']]))
            log.memory.info('Cur. predicted actions values: {}'.format(['{:.2f}'.format(x) for x in current_prob_dist]))
            log.memory.info('Best predicted actions values: {}'.format(['{:.2f}'.format(x) for x in best_prob_dist]))
            log.memory.info('')
