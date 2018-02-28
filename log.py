import pathlib

import paths
from log_utils import ConsoleLog, NullLog, create_logger

# Create log directory
pathlib.Path('{}logs'.format(paths.RUN)).mkdir(parents=True, exist_ok=True)

# Set all LOG_DISABLED to True to disable logging
# WARNING: the mcts log file gets big quite quickly
LOG_DISABLED = {
    'main': False, 'mcts': True, 'memory': True, 'model': True, 'tournament': False
}

main = create_logger('main', paths.RUN + 'logs/main.log')
main.disabled = LOG_DISABLED['main']

mcts = create_logger('mcts', paths.RUN + 'logs/mcts.log')
mcts.disabled = LOG_DISABLED['mcts']

memory = create_logger('memory', paths.RUN + 'logs/memory.log')
memory.disabled = LOG_DISABLED['memory']

model = create_logger('model', paths.RUN + 'logs/model.log')
model.disabled = LOG_DISABLED['model']

tournament = create_logger('tournament', paths.RUN + 'logs/tournament.log')
tournament.disabled = LOG_DISABLED['tournament']

null = NullLog()
console = ConsoleLog()
