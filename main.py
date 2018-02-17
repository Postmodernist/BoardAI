import os
import pickle
from shutil import copyfile

import numpy as np
from keras.utils import plot_model

import config
import initial
import paths
from agent import Agent
from game import Game
from loggers import log_main
from memory import Memory
from residual_cnn import ResidualCnn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
np.set_printoptions(suppress=True)


def main():
    log_main.info('=' * 80)
    log_main.info('New log')
    log_main.info('=' * 80)

    # Create game object
    env = Game()

    # Create an untrained neural network objects from the config file
    args_nn = (
        config.REG_CONST,  # regularization constant
        config.LEARNING_RATE,  # learning rate
        config.MOMENTUM,  # momentum
        env.input_shape,  # input dimensions
        env.action_size,  # output dimensions
        config.HIDDEN_CNN_LAYERS)  # hidden layers
    current_nn = ResidualCnn(*args_nn)
    best_nn = ResidualCnn(*args_nn)

    # Load an existing neural network
    if initial.RUN_NUMBER is not None:
        archive_base_path = '{}{}/run{0:0>4}/'.format(paths.RUN_ARCHIVE, env.name, initial.RUN_NUMBER)
        # Copy the config file to the root
        copyfile(archive_base_path + 'config.py', './config.py')
        # Load memories
        if initial.MEMORY_VERSION is not None:
            print('Loading memory version {}...'.format(initial.MEMORY_VERSION))
            memory_path = archive_base_path + 'memory/memory{0:0>4}.p'.format(initial.MEMORY_VERSION)
            memory = pickle.load(open(memory_path, "rb"))
        else:
            memory = Memory(config.MEMORY_SIZE)
        # Load neural network
        if initial.MODEL_VERSION is not None:
            best_player_version = initial.MODEL_VERSION
            print('Loading model version {}...'.format(best_player_version))
            model_path = '{}{}/run{0:0>4}/'.format(paths.RUN_ARCHIVE, env.name, initial.RUN_NUMBER)
            model_tmp = ResidualCnn.read(model_path, best_player_version)
            # Set the weights from the loaded model
            current_nn.model.set_weights(model_tmp.get_weights())
            best_nn.model.set_weights(model_tmp.get_weights())
        else:
            # Otherwise just ensure the weights on the two players are the same
            best_player_version = 0
            best_nn.model.set_weights(current_nn.model.get_weights())
        print()

    # Copy the config file to the run folder
    copyfile('./config.py', paths.RUN + 'config.py')

    # Plot model
    plot_model(current_nn.model, to_file=paths.RUN + 'models/model.png', show_shapes=True)

    # Create the players
    current_player = Agent('current_player', env.action_size, current_nn)
    best_player = Agent('best_player', env.action_size, best_nn)
    # human_player = HumanPlayer('human_player', env.action_size)


if __name__ == '__main__':
    main()
