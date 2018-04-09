import os
from pathlib import Path

from config import GAME
from utils.loaders import NeuralNet, Game
from utils.paths import ARCHIVE_DIR

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # suppress tf messages

DIR_NUMBER = 21
MODEL_VERSION = 7

# Load model
nnet = NeuralNet.create()
load_folder = str(Path(ARCHIVE_DIR, GAME, '{:04}'.format(DIR_NUMBER)))
load_name = 'model{:04}.h5'.format(MODEL_VERSION)
nnet.load(load_folder, load_name)

# Create state
state = Game.get_initial_state() \
    .get_next_state(3) \
    .get_next_state(10)

# Make a prediction
pi, value = nnet.predict(state.get_canonical_board(), state.get_valid_actions())

print("Pi:")
for i in range(7):
    print(" ".join("{:.2f}".format(x) for x in pi[i * 7:(i + 1) * 7]))

print("Value: " + str(value))
