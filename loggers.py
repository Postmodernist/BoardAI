import logging
from config import RUN_PATH


class ConsoleLog:
    @staticmethod
    def info(msg):
        print(msg)


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


# Set all LOG_DISABLED to True to disable logging
# WARNING: the mcts log file gets big quite quickly
LOG_DISABLED = {
    'main': False, 'mcts': False, 'memory': False, 'model': False, 'tournament': False
}

log_main = create_logger('main', RUN_PATH + 'logs/main.log')
log_main.disabled = LOG_DISABLED['main']

log_mcts = create_logger('mcts', RUN_PATH + 'logs/mcts.log')
log_mcts.disabled = LOG_DISABLED['mcts']

log_memory = create_logger('memory', RUN_PATH + 'logs/memory.log')
log_memory.disabled = LOG_DISABLED['memory']

log_model = create_logger('model', RUN_PATH + 'logs/model.log')
log_model.disabled = LOG_DISABLED['model']

log_tournament = create_logger('tournament', RUN_PATH + 'logs/tournament.log')
log_tournament.disabled = LOG_DISABLED['tournament']
