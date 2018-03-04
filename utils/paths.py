"""
Loading modules
---------------
Game path format: ./games/<game_name>/game.py
Model path format: ./games/<game_name>/<model_name>/neural_net.py

Loading data from archive
-------------------------
Memory path format: ./archive/<game_name>/####/memory####.pickle
Model path format: ./archive/<game_name>/####/model####.h5
"""

from pathlib import Path

from config import GAME, MODEL, LOAD_DIR_NUMBER, MEMORY_VERSION, MODEL_VERSION

TEMP_DIR = 'temp'
ARCHIVE_DIR = 'archive'

GAME_MODULE_PATH = 'games.{}.game'.format(GAME)
NNET_MODULE_PATH = 'games.{}.{}.neural_net'.format(GAME, MODEL)

# Path of memory object
MEMORY_FOLDER = str(Path(ARCHIVE_DIR, GAME, '{:04}'.format(LOAD_DIR_NUMBER or 0)))
MEMORY_NAME = 'memory{:04}.pickle'.format(MEMORY_VERSION or 0)

# Path of model weights
MODEL_FOLDER = str(Path(ARCHIVE_DIR, GAME, '{:04}'.format(LOAD_DIR_NUMBER or 0)))
MODEL_NAME = 'model{:04}.h5'.format(MODEL_VERSION or 0)

# Path of training losses plot
PLOT_LOSSES_FOLDER = TEMP_DIR
PLOT_LOSSES_NAME = 'train_losses.png'

# Path of logs
LOG_DIR = str(Path(TEMP_DIR, 'logs'))
