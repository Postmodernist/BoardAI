import logging

from game import Game


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


def create_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


def print_actions_prob_dist(logger, actions_prob_dist):
    """ Log actions probability distribution """
    n_rows = Game.board_shape[0]
    row_len = Game.board_shape[1]
    for row in range(n_rows):
        logger.info(['----' if prob == 0 else '{:.2f}'.format(prob)
                     for prob in actions_prob_dist[row_len * row: row_len * (row + 1)]])
