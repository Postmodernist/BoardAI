import os
import pathlib
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
from log_utils import print_actions_prob_dist, progress_bar
from memory import Memory
from model import ResidualCnn
from player import Player

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
np.set_printoptions(suppress=True)


def train_model():
    """ Self play / model retrain cycle """
    # Initialization
    print('\n----------------------------------------')
    # Create directories
    pathlib.Path('{}memory'.format(paths.RUN)).mkdir(parents=True, exist_ok=True)
    pathlib.Path('{}models'.format(paths.RUN)).mkdir(parents=True, exist_ok=True)
    pathlib.Path('{}plots'.format(paths.RUN)).mkdir(parents=True, exist_ok=True)
    # Load model
    current_nn, best_nn, best_player_version, memory = load_model()
    # Save current config file to run folder
    copyfile('./config.py', '{}config.py'.format(paths.RUN))
    # Plot model
    if not pathlib.Path('{}plots/model.png'.format(paths.RUN)).exists():
        print('Plotting model... ', end='')
        sys.stdout.flush()
        plot_model(current_nn.model, to_file='{}plots/model.png'.format(paths.RUN), show_shapes=True)
        print('done')
    # Create players
    print('Creating players... ', end='')
    sys.stdout.flush()
    current_player = Agent('Current_player', current_nn)
    best_player = Agent('Best_player', best_nn)
    print('done')
    print('----------------------------------------')
    # Collect memories
    if len(memory) < memory.size:
        print('\nCollecting memories... ')
        progress_bar(len(memory), memory.size)
        env = Game()
        while len(memory) < memory.size:
            play(env, best_player, best_player, config.TAU, memory)
            progress_bar(len(memory), memory.size)
        # Save memory
        memory.write('{}memory/memory{:04}.p'.format(paths.RUN, 0))
    log.main.info('')
    log.main.info('========================================')
    log.main.info('NEW RUN')
    log.main.info('========================================')
    iteration = 0
    while True:
        reload(log)
        reload(config)
        iteration += 1
        log.main.info('')
        log.main.info('----------------------------------------')
        log.main.info('Iteration: {}'.format(iteration))
        log.main.info('Best player version: {}'.format(best_player_version))
        log.main.info('----------------------------------------')
        print('\nIteration {}'.format(iteration))
        print('----------------------------------------')
        # Retrain the model
        best_player_version = retrain_model(current_player, best_player, best_player_version, memory)
        # Self play
        print('\nSelf play | Best player version: {}'.format(best_player_version))
        batch_play(config.EPISODES, best_player, best_player, config.TAU, memory, log.main)
        # Save memory
        if iteration % 5 == 0:
            memory.write('{}memory/memory{:04}.p'.format(paths.RUN, iteration))
            memory.log_sample(current_player, best_player)


def load_model():
    """ Load/create config, memory and model
        :rtype current_nn: ResidualCnn
        :rtype best_nn: ResidualCnn
        :rtype best_player_version: int
        :rtype memory: Memory
    """
    run_archive_path = ''
    # Load config
    if initial.RUN_NUMBER is not None:
        run_archive_path = '{}{}/run{:04}/'.format(paths.RUN_ARCHIVE, Game.name, initial.RUN_NUMBER)
        if pathlib.Path('{}config.py'.format(run_archive_path)).exists():
            copyfile('{}config.py'.format(run_archive_path), './config.py')
            reload(config)
    # Load memories
    if initial.RUN_NUMBER is not None and initial.MEMORY_VERSION is not None:
        memory = Memory.read('{}memory/memory{:04}.p'.format(run_archive_path, initial.MEMORY_VERSION))
    else:
        memory = Memory.create(config.MEMORY_SIZE)
    print('Creating untrained neural networks... ', end='')
    sys.stdout.flush()
    current_nn = ResidualCnn.create()
    best_nn = ResidualCnn.create()
    print('done')
    # Load model
    if initial.RUN_NUMBER is not None and initial.MODEL_VERSION is not None:
        best_player_version = initial.MODEL_VERSION
        model_tmp = ResidualCnn.read('{}models/version{:04}.h5'.format(run_archive_path, best_player_version))
        # Set the weights from the loaded model
        current_nn.model.set_weights(model_tmp.get_weights())
        best_nn.model.set_weights(model_tmp.get_weights())
    else:
        best_player_version = 0
        # Make sure both neural networks have the same weights
        print('Syncing models... ', end='')
        sys.stdout.flush()
        best_nn.model.set_weights(current_nn.model.get_weights())
        print('done')
    return current_nn, best_nn, best_player_version, memory


def play_custom(episodes: int, run_number: int, player1_ver: int, player2_ver: int, tau: int, logger=log.null):
    """ Play matches between specific versions of agents and/or human players
    :param episodes         Number of games to play
    :param run_number       Archived run ID to load model from
    :param player1_ver      Version of model for Player1, -1 for human player
    :param player2_ver      Version of model for Player2, -1 for human player
    :param tau              Number of turns before switching to deterministic play
    :param logger           Logger to write games progress and stats
    :return batch_play() return value
    """

    def create_player(ver, name):
        """ Create a human player or an agent """
        if ver == -1:
            # Create a human player
            player = HumanPlayer(name)
        else:
            # Create model for the player
            agent_nn = ResidualCnn.create()
            # Load model weights
            if ver > 0:
                path = '{}{}/run{:04}/models/version{:04}.h5'.format(paths.RUN_ARCHIVE, Game.name, run_number, ver)
                model_tmp = ResidualCnn.read(path)
                agent_nn.model.set_weights(model_tmp.get_weights())
            player = Agent(name, agent_nn)
        return player

    player1 = create_player(player1_ver, 'Player1')
    player2 = create_player(player2_ver, 'Player2')
    return batch_play(episodes, player1, player2, tau, None, logger, verbose=False)


def batch_play(episodes: int, player1: Player, player2: Player, tau: int, memory: Memory or None, logger=log.null,
               verbose=True):
    """ Play a batch of matches between player1 and player2
    :param episodes     Number of games to play
    :param player1      Player object
    :param player2      Player object
    :param tau          Turns before start playing deterministically
    :param memory       Memory object
    :param logger       Logger to write games progress and stats
    :param verbose      Write progress info to console
    :return wins        {player name: win count}
    """
    if verbose:
        print('Playing a batch of {} episodes...'.format(episodes))
    wins = {player1.name: 0, player2.name: 0, 'draw': 0}
    env = Game()
    for episode in range(episodes):
        logger.info('')
        logger.info('************ Episode {} of {} ************'.format(episode + 1, episodes))
        if verbose:
            progress_bar(episode + 1, episodes)
        # Shuffle players
        first = random.choice([1, -1])
        players = {first: player1, -first: player2}
        # Play a match
        state = play(env, players[1], players[-1], tau, memory, logger)
        # Update win counts
        if state.value == -1:
            winner = players[-state.player].name
            wins[winner] += 1
            logger.info('{} wins'.format(winner))
        else:
            wins['draw'] += 1
            logger.info('Game draw')
    return wins


def play(env: Game, player1: Player, player2: Player, tau: int, memory: Memory or None, logger=log.null):
    """ Play a single match between player1 and player2
    :param env      Game object
    :param player1  Player object
    :param player2  Player object
    :param tau      Turns before start playing deterministically
    :param memory   Memory object
    :param logger   Logger to write games progress and stats
    :return state   Final game state
    """
    # Reset game state
    state = env.reset()
    env.state.log(logger)
    # Clear MCTS trees
    player1.mcts = None
    player2.mcts = None
    turn = 0
    # Make moves until game is finished
    while not state.finished:
        turn += 1
        # Switch player
        player = player1 if state.player == 1 else player2
        logger.info('')
        logger.info('Turn: {} ({})'.format(player.name, env.pieces[state.player]))
        # Get player's action
        stochastic = turn < tau
        action, actions_prob_dist, mcts_value, nn_value = player.make_move(state, stochastic)
        # Update memory
        if memory is not None:
            identities = Game.identities(state, actions_prob_dist)
            memory.commit_short_memory(identities)
        # Write action to log
        if actions_prob_dist is not None:
            print_actions_prob_dist(logger, actions_prob_dist)
        action_msg = 'Action: {}'.format(action)
        if mcts_value is not None:
            action_msg = action_msg + ' | MCTS val {:.6f}'.format(mcts_value)
        if nn_value is not None:
            action_msg = action_msg + ' | RNN val {:.6f}'.format(nn_value)
        logger.info(action_msg)
        # Make the move
        state = env.make_move(action)
        env.state.log(logger)
    # Commit new memories
    if memory is not None:
        # Assign state values to memory items
        for item in memory.short_memory:
            if item['state'].player == state.player:
                item['value'] = state.value
            else:
                item['value'] = -state.value
        memory.commit_long_memory()
    return state


def retrain_model(current_player: Player, best_player: Player, best_player_version: int, memory: Memory):
    """ Retrain model
    :param current_player           Copy of the agent being retrained
    :param best_player              The agent
    :param best_player_version      Version of the agent
    :param memory                   Memory object
    :return best_player_version     Updated version of the agent
    """
    current_nn = current_player.nn
    best_nn = best_player.nn
    log.model.info('Retraining model...')
    print('\nRetraining model...')
    sys.stdout.flush()
    current_player.nn.retrain(memory.long_memory)
    # Tournament
    print('\nTournament')
    sys.stdout.flush()
    wins = batch_play(config.EVAL_EPISODES, best_player, current_player, 0, None, log.tournament)
    print('Wins: {}'.format(wins))
    total_wins = wins[current_player.name] + wins[best_player.name]
    current_player_wins_ratio = wins[current_player.name] / total_wins if total_wins != 0 else 0
    if current_player_wins_ratio > config.WINS_RATIO:
        best_player_version += 1
        print('Setting up new best model... ', end='')
        sys.stdout.flush()
        best_nn.model.set_weights(current_nn.model.get_weights())
        print('done')
        best_nn.write('{}models/version{:04}.h5'.format(paths.RUN, best_player_version))
    return best_player_version


def test_predictions(run_number, model_ver):
    """ Load model and make a sample prediction """
    # Create a new game and make some moves
    env = Game()
    env.make_move(random.choice(env.state.allowed_actions))
    env.make_move(random.choice(env.state.allowed_actions))
    env.make_move(random.choice(env.state.allowed_actions))
    env.make_move(random.choice(env.state.allowed_actions))
    env.make_move(random.choice(env.state.allowed_actions))
    print(env.state)
    print('Creating model... ', end='')
    sys.stdout.flush()
    nn = ResidualCnn.create()
    path = '{}{}/run{:04}/models/version{:04}.h5'.format(paths.RUN_ARCHIVE, Game.name, run_number, model_ver)
    model_tmp = ResidualCnn.read(path, verbose=False)
    nn.model.set_weights(model_tmp.get_weights())
    print('done')
    print('Making predictions... ', end='')
    sys.stdout.flush()
    value, pi = nn.predict(env.state)
    print('done')
    print('value = ' + str(value))
    print(np.round(pi.reshape(Game.board_shape), 4))


def test_play():
    """ Play custom set of games with chosen opponents """
    wins = play_custom(
        episodes=1,  # number of games
        run_number=0,
        player1_ver=-1,  # -1 for human player
        player2_ver=-1,  # -1 for human player
        tau=0,
        logger=log.console)
    print('Wins: {}'.format(wins))


if __name__ == '__main__':
    train_model()
    # test_predictions(0, 1)
    # test_play()
