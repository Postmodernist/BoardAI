import pickle
from collections import deque
from pathlib import Path
from sys import stdout


class Memory:

    def __init__(self, size: int):
        self.size = size
        self.short = []
        self.long = deque(maxlen=size)

    def __len__(self):
        return len(self.long)

    @staticmethod
    def create(size: int):
        print('Creating new memory... ', end='')
        stdout.flush()
        memory = Memory(size)
        print('done')
        return memory

    def append(self, item: tuple):
        """ Append data to short memory
        :param item: tuple (canonical_board, pi, player, None) items
        """
        self.short.append(item)

    def commit(self):
        """ Move data from short to long memory """
        self.long.extend(self.short)
        self.short = []

    def save(self, folder: str, name: str):
        print('Saving memory to {} ... '.format(name), end='')
        stdout.flush()
        Path(folder).mkdir(parents=True, exist_ok=True)
        path = Path(folder, name)
        pickle.dump(self.long, open(str(path), 'wb'))
        print('done')

    def load(self, folder: str, name: str):
        print('Loading memory from {} ... '.format(name), end='')
        stdout.flush()
        path = Path(folder, name)
        if not path.exists():
            print()
            raise "File not found: {}".format(path)
        long_memory = pickle.load(open(str(path), 'rb'))
        if self.size < long_memory.maxlen:
            print(' memory size is too big, trimming... ')
        self.long.clear()
        self.long.extend(long_memory)
        print('done')
