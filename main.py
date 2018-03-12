import os
import random
import sys
from importlib import reload
from pathlib import Path
from shutil import copyfile
from sys import stdout

from keras.utils import plot_model

import config
from evaluate import Evaluate
from players import Human, ClassicMctsAgent, MctsAgent, build_player
from self_play import SelfPlay
from utils.loaders import load_memory, load_model
from utils.loggers import console as console_log
from utils.paths import TEMP_DIR

# Run mode -- 0: learn, 1: custom play
if len(sys.argv) > 1:
    RUN_MODE = int(sys.argv[1])
else:
    RUN_MODE = 0

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # suppress tf messages


def learn():
    """
    Perform iterations with SELF_PLAY_EPISODES of self-play in each
    iteration. After every iteration, it retrains neural network with
    examples in memory (which has a maximum length of MEMORY_SIZE).
    It then pits the new neural network against the old one and accepts it
    only if it wins >= EVAL_THRESHOLD fraction of games.
    """
    # Save current config file to temp folder
    Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)
    copyfile('config.py', str(Path(TEMP_DIR, 'config.py')))

    # Initialize memory and model
    memory = load_memory()
    best_nn, adversary_nn, model_version = load_model()

    print('Plotting model... ', end='')
    stdout.flush()
    plot_model(best_nn.model, to_file=str(Path(TEMP_DIR, 'model.png')), show_shapes=True)
    print('done')

    print('Creating agents... ', end='')
    stdout.flush()
    best_agent = MctsAgent('best_agent', best_nn, config.MCTS_TRAIN_SIMULATIONS, config.STOCHASTIC_TURNS)
    adversary_agent = MctsAgent('adversary_agent', adversary_nn, config.MCTS_COMPETITIVE_SIMULATIONS, 0)
    print('done')

    self_play = SelfPlay(best_agent)
    evaluate = Evaluate(best_agent, adversary_agent)
    iteration = 0

    while True:
        reload(config)
        # Self-play
        iteration += 1
        best_agent.set_simulations(config.MCTS_TRAIN_SIMULATIONS)
        best_agent.set_pi_turns(config.STOCHASTIC_TURNS)
        self_play.batch(config.SELF_PLAY_EPISODES, memory, 'Self-play (iter {})'.format(iteration))
        memory.save(TEMP_DIR, 'memory{:04}.pickle'.format(iteration))
        # Train
        examples = list(memory.long)
        random.shuffle(examples)
        adversary_nn.train(examples)
        # Evaluate
        best_agent.set_simulations(config.MCTS_COMPETITIVE_SIMULATIONS)
        best_agent.set_pi_turns(0)
        wins = evaluate.batch(config.EVAL_EPISODES, 'Evaluate (model {})'.format(model_version))
        best_wins = wins[best_agent.get_name()]
        adversary_wins = wins[adversary_agent.get_name()]
        total_wins = best_wins + adversary_wins
        if total_wins == 0 or adversary_wins / total_wins < config.EVAL_THRESHOLD:
            print('New model REJECTED')
            adversary_nn.model.set_weights(best_nn.model.get_weights())
        else:
            print('New model ACCEPTED')
            best_nn.model.set_weights(adversary_nn.model.get_weights())
            model_version += 1
            best_nn.save(TEMP_DIR, 'model{:04}.h5'.format(model_version))


def play_custom():
    """ Play versus agent or pit two agents against each other """
    episodes = 40
    player0 = build_player('Human', Human)
    player1 = build_player('Botik', ClassicMctsAgent, simulations=10000, pi_turns=0, verbose=True)
    player2 = build_player('AI-1600', MctsAgent, dir_number=10, model_version=5, simulations=500, pi_turns=0, verbose=True)
    player3 = build_player('AI-500', MctsAgent, dir_number=1, model_version=6, simulations=500, pi_turns=0, verbose=True)
    Evaluate(player2, player3, console_log, verbose=False).batch(episodes, 'Custom matchup')


if __name__ == '__main__':
    if RUN_MODE == 0:
        learn()
    else:
        play_custom()
