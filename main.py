import os
import numpy as np
import config
from loggers import log_main
from game import Game
from model import ResidualCnn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
np.set_printoptions(suppress=True)

log_main.info('=' * 80)
log_main.info('New log')
log_main.info('=' * 80)

# Create game object
env = Game()

# Create an untrained neural network objects from the config file
params = (
    config.REG_CONST,  # regularization constant
    config.LEARNING_RATE,  # learning rate
    env.input_shape,  # input dimensions
    env.action_size,  # output dimensions
    config.HIDDEN_CNN_LAYERS)  # hidden layers
current_nn = ResidualCnn(*params)
best_nn = ResidualCnn(*params)
