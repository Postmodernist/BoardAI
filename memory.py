from collections import deque


class Memory:

    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.long_memory = deque(maxlen=memory_size)
        self.short_memory = deque(maxlen=memory_size)

    def commit_short_memory(self, identities):
        for identity in identities:
            state = identity[0]
            action_values = identity[1]
            item = {
                'state': state,
                'action_values': action_values,
                'value': None}
            self.short_memory.append(item)

    def commit_long_memory(self):
        self.long_memory.extend(self.short_memory)
        self.short_memory.clear()
