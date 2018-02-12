from collections import deque


class Memory:

    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.long_memory = deque(maxlen=memory_size)
        self.short_memory = deque(maxlen=memory_size)

    def commit_short_memory(self, identities, state, action_values):
        for x in identities(state, action_values):
            item = {'board': x[0].board, 'state': x[0], 'id': x[0].id, 'action_values': x[1], 'player': x[0].player}
            self.short_memory.append(item)

    def commit_long_memory(self):
        for x in self.short_memory:
            self.long_memory.append(x)
        self.clear_short_memory()

    def clear_short_memory(self):
        self.short_memory = deque(maxlen=self.memory_size)
