import logging
from pathlib import Path

from config import LOG_DISABLED
from utils.paths import LOG_DIR


class ConsoleLog:

    def __init__(self):
        self.disabled = False

    def info(self, *args):
        if self.disabled:
            return
        if len(args) < 2:
            print(*args)
        else:
            fmt = args[0]
            idx = fmt.find('%')
            while idx != -1:
                fmt = fmt.replace(fmt[idx:idx + 2], '{}')
                idx = fmt.find('%')
            print(fmt.format(*args[1:]))


class NullLog:

    def __init__(self):
        self.disabled = True

    def info(self, *args):
        pass


def create_logger(name, path, disabled):
    formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.FileHandler(path)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(handler)
    logger.disabled = disabled
    return logger


# Create log directory
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

# Create loggers
training = create_logger('train', str(Path(LOG_DIR, 'train.log')), LOG_DISABLED['train'])
evaluation = create_logger('eval', str(Path(LOG_DIR, 'eval.log')), LOG_DISABLED['eval'])
mcts = create_logger('mcts', str(Path(LOG_DIR, 'mcts.log')), LOG_DISABLED['mcts'])
console = ConsoleLog()
null = NullLog()
