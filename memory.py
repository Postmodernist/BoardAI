import pickle
import random
import sys
from collections import deque

import log
from player import Player


class Memory:

    def __init__(self, size: int):
        self.size = size
        self.long_memory = deque(maxlen=size)
        self.short_memory = deque(maxlen=size)

    def __len__(self):
        return len(self.long_memory)

    def commit_short_memory(self, identities: list):
        for identity in identities:
            item = {
                'state': identity[0],
                'actions_prob_dist': identity[1],
                'value': None}
            self.short_memory.append(item)

    def commit_long_memory(self):
        self.long_memory.extend(self.short_memory)
        self.short_memory.clear()

    @staticmethod
    def create(size: int, verbose=True):
        if verbose:
            print('Initializing memory... ', end='')
            sys.stdout.flush()
        memory = Memory(size)
        if verbose:
            print('done')
        return memory

    @staticmethod
    def read(path: str, verbose=True):
        if verbose:
            print('Reading memory... ', end='')
            sys.stdout.flush()
        memory = pickle.load(open(path, 'rb'))
        if verbose:
            print('done')
        # Convert old memory format
        try:
            memory.size
        except AttributeError:
            memory_tmp = Memory(len(memory.long_memory))
            memory_tmp.long_memory = memory.long_memory
            return memory_tmp
        return memory

    def write(self, path: str, verbose=True):
        if verbose:
            print('Writing memory... ', end='')
            sys.stdout.flush()
        pickle.dump(self, open(path, 'wb'))
        if verbose:
            print('done')

    def log_sample(self, current_player: Player, best_player: Player, verbose=True):
        if log.memory.disabled:
            return
        if verbose:
            print('Writing memory sample to log... ', end='')
            sys.stdout.flush()
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
                'MCTS simulated P(action): {}'.format(['{:.2f}'.format(x) for x in mem['actions_prob_dist']]))
            log.memory.info('Cur. predicted P(action): {}'.format(['{:.2f}'.format(x) for x in current_prob_dist]))
            log.memory.info('Best predicted P(action): {}'.format(['{:.2f}'.format(x) for x in best_prob_dist]))
            log.memory.info('')
        if verbose:
            print('done')
