import os
import pickle
import random
from importlib import reload
from shutil import copyfile

import numpy as np
from keras.utils import plot_model

import config
import initial
import paths
from agent import Agent
from game import Game
from human_player import HumanPlayer
from loggers import log_main, log_memory, log_tournament
from memory import Memory
from residual_cnn import ResidualCnn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
np.set_printoptions(suppress=True)


def main():
    log_main.info('=' * 40)
    log_main.info('New log')
    log_main.info('=' * 40)

    # Create an untrained neural network objects from the config file
    env = Game()
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
    if initial.RUN_NUMBER is not None and initial.MEMORY_VERSION is not None and initial.MODEL_VERSION is not None:
        archive_base_path = '{}{}/run{0:0>4}/'.format(paths.RUN_ARCHIVE, env.name, initial.RUN_NUMBER)
        # Copy the config file to the root
        copyfile(archive_base_path + 'config.py', './config.py')
        # Load memories
        print('Loading memory version {}...'.format(initial.MEMORY_VERSION))
        memory_path = archive_base_path + 'memory/memory{0:0>4}.p'.format(initial.MEMORY_VERSION)
        memory = pickle.load(open(memory_path, 'rb'))
        # Load neural network
        best_player_version = initial.MODEL_VERSION
        print('Loading model version {}...'.format(best_player_version))
        model_path = '{}{}/run{0:0>4}/'.format(paths.RUN_ARCHIVE, env.name, initial.RUN_NUMBER)
        model_tmp = ResidualCnn.read(model_path, best_player_version)
        # Set the weights from the loaded model
        current_nn.model.set_weights(model_tmp.get_weights())
        best_nn.model.set_weights(model_tmp.get_weights())
        print()
    # Cold start
    else:
        memory = Memory(config.MEMORY_SIZE)
        best_player_version = 0
        # Make sure both neural networks have the same weights
        best_nn.model.set_weights(current_nn.model.get_weights())

    # Copy the config file to the run folder
    copyfile('./config.py', paths.RUN + 'config.py')

    # Plot model
    plot_model(current_nn.model, to_file=paths.RUN + 'models/model.png', show_shapes=True)

    # Create the players
    current_player = Agent('current_player', env.action_size, current_nn)
    best_player = Agent('best_player', env.action_size, best_nn)
    # human_player = HumanPlayer('human_player', env.action_size)

    iteration = 0
    while True:
        iteration += 1
        reload(config)
        # Self play
        log_main.info('Best player version: %d', best_player_version)
        print('Iteration number: ' + str(iteration))
        print('Best player version: ' + str(best_player_version))
        print('Self playing ' + str(config.EPISODES) + ' episodes...')
        _, _, _, memory = play_matches(
            player1=best_player,
            player2=best_player,
            episodes=config.EPISODES,
            log=log_main,
            stochastic_turns=config.STOCHASTIC_TURNS,
            memory=memory)
        print()
        memory.short_memory.clear()
        if len(memory.long_memory) >= config.MEMORY_SIZE:
            # Retrain the neural net
            print('Retraining...')
            current_player.nn.retrain(memory.long_memory)
            # Save memory
            if iteration % 5 == 0:
                memory_path = paths.RUN + 'memory/memory{0:0>4}.p'.format(iteration)
                pickle.dump(memory, open(memory_path, 'wb'))
            # Log a random memory sample of size <1000
            log_memory.info('=' * 40)
            log_memory.info('New memories')
            log_memory.info('=' * 40)
            memory_sample = random.sample(memory.long_memory, min(1000, len(memory.long_memory)))
            for mem in memory_sample:
                current_value, current_prob_dist = current_player.mcts.get_nn_predictions(mem['state'])
                best_value, best_prob_dist = best_player.mcts.get_nn_predictions(mem['state'])
                log_memory.info('MCTS value for player %s: %f', mem['player'], mem['value'])
                log_memory.info('Current predicted value for player %s: %f', mem['state'].player, current_value)
                log_memory.info('Best predicted value for player %s: %f', mem['state'].player, best_value)
                log_memory.info('          MCTS action values: %s', ['{:.2f}'.format(x) for x in mem['action_values']])
                log_memory.info('Cur. predicted action values: %s', ['{:.2f}'.format(x) for x in current_prob_dist])
                log_memory.info('Best predicted action values: %s', ['{:.2f}'.format(x) for x in best_prob_dist])
                log_memory.info('ID: %s', mem['state'].id)
                log_memory.info('Input to model: %s', current_player.nn.state_to_model_input(mem['state']))
                log_memory.info(mem['state'])
            # Tournament
            print('Playing tournament...')
            win_counts, sp_win_counts, score_history, _ = play_matches(
                player1=best_player,
                player2=current_player,
                episodes=config.EVAL_EPISODES,
                log=log_tournament,
                stochastic_turns=0,
                memory=None)
            print('Win counts')
            print(win_counts)
            print('Starting / non-starting players win counts')
            print(sp_win_counts)
            print('Scores')
            print(best_player.name + ': ' + str(sum(score_history[best_player.name])))
            print(current_player.name + ': ' + str(sum(score_history[current_player.name])))
            print('\n')
            if win_counts[current_player.name] > win_counts[best_player.name] * config.SCORING_THRESHOLD:
                best_player_version += 1
                best_nn.model.set_weights(current_nn.model.get_weights())
                best_nn.write(best_player_version)
        else:
            print('Memory size: ' + str(len(memory.long_memory)))


def play_matches_between_versions(env, run_number, player1_ver, player2_ver, episodes, log, stochastic_turns):
    """ Play matches between specific versions of agents and/or human players """

    def create_player(version, name):
        if version == -1:
            # Create a human player
            player = HumanPlayer(name, env.action_size)
        else:
            # Create neural net for player1
            agent_nn = ResidualCnn(
                config.REG_CONST,
                config.LEARNING_RATE,
                config.MOMENTUM,
                env.input_shape,
                env.action_size,
                config.HIDDEN_CNN_LAYERS)
            # Load model weights
            if version > 0:
                model_path = '{}{}/run{0:0>4}/'.format(paths.RUN_ARCHIVE, env.name, run_number)
                model_tmp = ResidualCnn.read(model_path, version)
                agent_nn.model.set_weights(model_tmp.get_weights())
            player = Agent(name, env.action_size, agent_nn)
        return player

    player1 = create_player(player1_ver, 'player1')
    player2 = create_player(player2_ver, 'player2')
    return play_matches(player1, player2, episodes, log, stochastic_turns, memory=None)


def play_matches(player1, player2, episodes, log, stochastic_turns, memory):
    """ Play matches between player1 and player2 """
    env = Game()
    win_counts = {player1.name: 0, player2.name: 0, 'draw': 0}
    sp_win_counts = {'sp': 0, 'nsp': 0, 'draw': 0}  # starting / non-starting players
    score_history = {player1.name: [], player2.name: []}  # win is +1 point, loss is -1 point, draw is 0

    for episode in range(episodes):
        log.info('::: Episode %d of %d :::', episode + 1, episodes)
        print(episode + 1, end=' ')
        player1.mcts = None
        player2.mcts = None
        first_player = random.choice([1, -1])
        if first_player == 1:
            players = {1: player1, -1: player2}
            log.info('%s goes first', player1.name)
        else:
            players = {-1: player1, 1: player2}
            log.info('%s goes first', player2.name)
        state = env.reset()
        log.info(env.state)
        turn = 0
        finished = False

        while not finished:
            turn += 1
            # Run the MCTS and return an action
            stochastic = turn < stochastic_turns
            action, actions_prob_dist, mcts_value, nn_value = players[state.player].make_move(state, stochastic)
            if memory is not None:
                # Commit the move to memory
                identities = env.identities(state, actions_prob_dist)
                memory.commit_short_memory(identities)
            # Log actions probability distribution
            log.info('Action: %d', action)
            n_rows = env.board_shape[0]
            row_len = env.board_shape[1]
            for row in range(n_rows):
                log.info(['----' if prob == 0 else '{:.2f}'.format(prob)
                          for prob in actions_prob_dist[row_len * row: row_len * (row + 1)]])
            log.info('MCTS perceived value for %s: %f', state.pieces[state.player], np.round(mcts_value, 2))
            log.info('NN perceived value for %s: %f', state.pieces[state.player], np.round(nn_value, 2))
            # Make the move. The value of the new state from the POV of the new player, i.e. -1 if
            # the previous player played a winning move or 0 otherwise
            state, value, finished = env.make_move(action)
            log.info(env.state)

        # Assign the values correctly to the game moves
        if memory is not None:
            for move in memory.short_memory:
                if move['state'].player == state.player:
                    move['value'] = value
                else:
                    move['value'] = -value
            memory.commit_long_memory()

        # Update win counters and scores
        if value == -1:
            log.info('%s wins', players[-state.player].name)
            win_counts[players[-state.player].name] += 1
            if state.player == 1:
                sp_win_counts['nsp'] += 1
            else:
                sp_win_counts['sp'] += 1
        else:
            log.info('Game draw')
            win_counts['draw'] += 1
            sp_win_counts['draw'] += 1
        score_history[players[state.player].name].append(state.score[0])
        score_history[players[-state.player].name].append(state.score[1])
        log.info('-' * 20)

    return win_counts, sp_win_counts, score_history, memory


if __name__ == '__main__':
    main()
