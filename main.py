import os
import pathlib
import pickle
import random
import sys
from importlib import reload
from shutil import copyfile

import numpy as np
from keras.utils import plot_model

import config
import initial
import log
import paths
from agent import Agent
from game import Game
from human_player import HumanPlayer
from log_utils import print_actions_prob_dist
from memory import Memory
from residual_cnn import ResidualCnn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
np.set_printoptions(suppress=True)


def test_play():
    """ Play custom set of games with chosen opponents """
    res = play_custom(
        run_number=0,
        player1_ver=-1,  # -1 for human player
        player2_ver=-1,  # -1 for human player
        episodes=1,  # number of games
        logger=log.console,
        stochastic_turns=0)
    print(res[:3])


def train_model():
    """ Create two identical agents and train model by self playing """
    print('\nInitialization')
    print('----------------------------------------')
    # Create directory structure
    pathlib.Path('{}memory'.format(paths.RUN)).mkdir(parents=True, exist_ok=True)
    pathlib.Path('{}models'.format(paths.RUN)).mkdir(parents=True, exist_ok=True)
    pathlib.Path('{}plots'.format(paths.RUN)).mkdir(parents=True, exist_ok=True)
    # Load model
    current_nn, best_nn, best_player_version, memory = load_model()
    # Save current config file to run folder
    copyfile('./config.py', '{}config.py'.format(paths.RUN))
    # Plot model
    if config.PLOT_MODEL_GRAPH:
        print('Plotting model... ', end='')
        sys.stdout.flush()
        plot_model(current_nn.model, to_file='{}models/model.png'.format(paths.RUN), show_shapes=True)
        print('done')
    # Create players
    print('Creating players... ', end='')
    sys.stdout.flush()
    current_player = Agent('Current_player', Game.board_size, current_nn)
    best_player = Agent('Best_player', Game.board_size, best_nn)
    print('done')
    print('----------------------------------------')
    log.main.info('')
    log.main.info('========================================')
    log.main.info('Start iterations of self play')
    log.main.info('========================================')
    print('\nStart iterations of self play')
    iteration = 0
    while True:
        iteration += 1
        reload(log)
        reload(config)
        if config.STOP_TRAINING:
            break
        log.main.info('')
        log.main.info('----------------------------------------')
        log.main.info('Iteration {}'.format(iteration))
        log.main.info('Best player version: {}'.format(best_player_version))
        log.main.info('----------------------------------------')
        print('\nIteration {}'.format(iteration))
        print('Best player version: {}'.format(best_player_version))
        _, _, _, memory = play(
            player1=best_player,
            player2=best_player,
            episodes=config.EPISODES,
            logger=log.main,
            stochastic_turns=config.STOCHASTIC_TURNS,
            memory=memory)
        memory.short_memory.clear()
        print('Memory size: {}'.format(len(memory.long_memory)))
        if len(memory.long_memory) >= config.MEMORY_SIZE:
            # Retrain the model
            best_player_version = retrain_model(current_player, best_player, iteration, best_player_version, memory)


def load_model():
    # Create untrained neural network objects
    print('Creating untrained neural networks... ', end='')
    sys.stdout.flush()
    args = (
        config.REG_CONST,  # regularization constant
        config.LEARNING_RATE,  # learning rate
        config.MOMENTUM,  # momentum
        Game.input_shape,  # input dimensions
        Game.board_size,  # output dimensions
        config.HIDDEN_CNN_LAYERS)  # hidden layers
    current_nn = ResidualCnn(*args)
    best_nn = ResidualCnn(*args)
    print('done')
    # Cold start
    if initial.RUN_NUMBER is None or initial.MEMORY_VERSION is None or initial.MODEL_VERSION is None:
        print('Initializing memory... ', end='')
        sys.stdout.flush()
        memory = Memory(config.MEMORY_SIZE)
        print('done')
        best_player_version = 0
        # Make sure both neural networks have the same weights
        print('Syncing models... ', end='')
        sys.stdout.flush()
        best_nn.model.set_weights(current_nn.model.get_weights())
        print('done')
    # Load an existing neural network
    else:
        model_path = '{}{}/run{:04}/'.format(paths.RUN_ARCHIVE, Game.name, initial.RUN_NUMBER)
        # Load the config file from the archived run folder
        copyfile('{}config.py'.format(model_path), './config.py')
        # Load memories
        print('Loading memory version {}... '.format(initial.MEMORY_VERSION), end='')
        sys.stdout.flush()
        memory_path = '{}memory/memory{:04}.p'.format(model_path, initial.MEMORY_VERSION)
        memory = pickle.load(open(memory_path, 'rb'))
        print('done')
        # Load neural network
        best_player_version = initial.MODEL_VERSION
        print('Loading model version {}... '.format(best_player_version), end='')
        sys.stdout.flush()
        model_tmp = ResidualCnn.read(model_path, best_player_version)
        # Set the weights from the loaded model
        current_nn.model.set_weights(model_tmp.get_weights())
        best_nn.model.set_weights(model_tmp.get_weights())
        print('done')
    return current_nn, best_nn, best_player_version, memory


def play_custom(run_number, player1_ver, player2_ver, episodes, logger, stochastic_turns):
    """ Play matches between specific versions of agents and/or human players
    :param run_number           Archived run ID to load model from
    :param player1_ver          Version of model for Player1, -1 for human player
    :param player2_ver          Version of model for Player2, -1 for human player
    :param episodes             Number of games to play
    :param logger               Logger to write games progress and stats
    :param stochastic_turns     Number of stochastic turns to take before switching to deterministic
    :return play() return value
    """

    def create_player(version, name):
        """ Create a human player or an agent """
        if version == -1:
            # Create a human player
            player = HumanPlayer(name, Game.board_size)
        else:
            # Create neural net for player1
            agent_nn = ResidualCnn(
                config.REG_CONST,
                config.LEARNING_RATE,
                config.MOMENTUM,
                Game.input_shape,
                Game.board_size,
                config.HIDDEN_CNN_LAYERS)
            # Load model weights
            if version > 0:
                model_path = '{}{}/run{:04}/'.format(paths.RUN_ARCHIVE, Game.name, run_number)
                model_tmp = ResidualCnn.read(model_path, version)
                agent_nn.model.set_weights(model_tmp.get_weights())
            player = Agent(name, Game.board_size, agent_nn)
        return player

    player1 = create_player(player1_ver, 'Player1')
    player2 = create_player(player2_ver, 'Player2')
    return play(player1, player2, episodes, logger, stochastic_turns, memory=None)


def play(player1, player2, episodes, logger, stochastic_turns, memory):
    """ Play a set of matches between player1 and player2
    :param player1              Player object
    :param player2              Player object
    :param episodes             Number of games to play
    :param logger               Logger to write games progress and stats
    :param stochastic_turns     Number of stochastic turns to take before switching to deterministic
    :param memory               Memory object
    :return win_counts          {player name: win count}
    :return sns_win_counts      {starting / non-starting player: win count}
    :return score_history       {player name: list of score changes after each game}
    :return memory              Updated memory object
    """

    env = Game()
    win_counts = {player1.name: 0, player2.name: 0, 'draw': 0}
    sns_win_counts = {'sp': 0, 'nsp': 0, 'draw': 0}  # starting / non-starting players
    score_history = {player1.name: [], player2.name: []}  # win is +1 point, loss is -1 point, draw is 0

    for episode in range(episodes):
        logger.info('')
        logger.info('************ Episode {} of {} ************'.format(episode + 1, episodes))
        print('\rEpisode {} of {}'.format(episode + 1, episodes), end='')
        player1.mcts = None
        player2.mcts = None
        first = random.choice([1, -1])
        players = {first: player1, -first: player2}
        state = env.reset()
        turn = 0

        while not state.finished:
            logger.info('')
            logger.info('{}\'s turn'.format(players[state.player].name))
            turn += 1
            # Get player's action
            stochastic = turn < stochastic_turns
            action, actions_prob_dist, mcts_value, nn_value = players[state.player].make_move(state, stochastic)
            if memory is not None:
                # Commit the move to memory
                identities = env.identities(state, actions_prob_dist)
                memory.commit_short_memory(identities)
            if actions_prob_dist is not None:
                print_actions_prob_dist(logger, actions_prob_dist)
            if mcts_value is not None:
                logger.info('MCTS value for {}: {:.6f}'.format(Game.pieces[state.player], mcts_value))
            if nn_value is not None:
                logger.info('RNN  value for {}: {:.6f}'.format(Game.pieces[state.player], nn_value))
            # Make the move. The value of the new state from the POV of the new player, i.e. -1 if
            # the previous player played a winning move or 0 otherwise
            state = env.make_move(action)
            logger.info('Action: {}'.format(action))
            env.state.log(logger)

        # Update memory
        if memory is not None:
            # Assign the values correctly to the game moves
            for move in memory.short_memory:
                if move['state'].player == state.player:
                    move['value'] = state.value
                else:
                    move['value'] = -state.value
            memory.commit_long_memory()

        # Update win counters and scores
        if state.value == -1:
            logger.info('{} wins'.format(players[-state.player].name))
            win_counts[players[-state.player].name] += 1
            if state.player == 1:
                sns_win_counts['nsp'] += 1
            else:
                sns_win_counts['sp'] += 1
        else:
            logger.info('Game draw')
            win_counts['draw'] += 1
            sns_win_counts['draw'] += 1
        score_history[players[state.player].name].append(state.score[0])
        score_history[players[-state.player].name].append(state.score[1])
    print()
    return win_counts, sns_win_counts, score_history, memory


def retrain_model(current_player, best_player, iteration, best_player_version, memory):
    """ Retrain model """
    current_nn = current_player.nn
    best_nn = best_player.nn
    log.model.info('Retraining model...')
    print('\nRetraining model...')
    sys.stdout.flush()
    current_player.nn.retrain(memory.long_memory)
    if iteration % 1 == 0:
        print('\nSaving memory... ', end='')
        sys.stdout.flush()
        memory_path = '{}memory/memory{:04}.p'.format(paths.RUN, iteration)
        pickle.dump(memory, open(memory_path, 'wb'))
        print('done')
    print('\nWriting memory sample to log... ', end='')
    sys.stdout.flush()
    memory.log_long_memory_sample(current_player, best_player)
    print('done')
    # Tournament
    print('\nPlaying tournament...')
    sys.stdout.flush()
    win_counts, sns_win_counts, score_history, _ = play(
        player1=best_player,
        player2=current_player,
        episodes=config.EVAL_EPISODES,
        logger=log.tournament,
        stochastic_turns=0,
        memory=None)
    print('Win counts: {}'.format(win_counts))
    print('(Non-)starting win counts: {}'.format(sns_win_counts))
    if win_counts[current_player.name] > win_counts[best_player.name] * config.SCORING_THRESHOLD:
        best_player_version += 1
        print('\nSetting up new best model... ', end='')
        sys.stdout.flush()
        best_nn.model.set_weights(current_nn.model.get_weights())
        print('done')
        print('Saving model... ', end='')
        sys.stdout.flush()
        best_nn.write(best_player_version)
        print('done')
    return best_player_version


if __name__ == '__main__':
    train_model()
    # test_play()
