import random
from pathlib import Path

import numpy as np

from config import GAME
from intefraces.i_game_state import IGameState
from intefraces.i_neural_net import INeuralNet
from intefraces.i_player import IPlayer
from mcts import Mcts
from mcts_classic import MctsClassic
from utils.loaders import Game, NeuralNet
from utils.paths import ARCHIVE_DIR


class Human(IPlayer):

    def __init__(self, name: str):
        self._name = name

    def get_name(self) -> str:
        return self._name

    def get_action(self, state: IGameState) -> (int, np.ndarray):
        valid_actions = state.get_valid_actions()
        print(valid_actions)
        while True:
            action = int(input(': '))
            if action in valid_actions:
                return action, None
            print('Invalid action')


class ClassicMctsAgent(IPlayer):

    def __init__(self, name: str, simulations: int, pi_turns: int = 0, verbose: bool = True):
        self._name = name
        self._simulations = simulations
        self._pi_turns = pi_turns
        self._verbose = verbose
        self._mcts = MctsClassic()

    def get_name(self):
        return self._name

    def get_action(self, state: IGameState) -> (int, np.ndarray):
        pi = self._mcts.get_distribution(state, self._simulations, self._verbose)
        if state.get_turn() >= self._pi_turns:
            # Choose action deterministically
            candidates = np.nonzero(pi == max(pi))[0]
            action = random.choice(candidates)  # tie break
        else:
            # Choose action stochastically
            action = np.random.choice(np.arange(Game.ACTION_SIZE), p=pi)
        return action, pi

    def set_pi_turns(self, pi_turns: int):
        self._pi_turns = pi_turns


class MctsAgent(IPlayer):

    def __init__(self, name: str, nnet: INeuralNet, simulations: int, pi_turns: int, verbose: bool = False):
        self._name = name
        self._simulations = simulations
        self._pi_turns = pi_turns
        self._verbose = verbose
        self._mcts = Mcts(nnet)

    def get_name(self):
        return self._name

    def get_action(self, state) -> (int, np.ndarray):
        pi = self._mcts.get_distribution(state, self._simulations, self._verbose)
        if state.get_turn() >= self._pi_turns:
            # Choose action deterministically
            candidates = np.nonzero(pi == max(pi))[0]
            action = random.choice(candidates)  # tie break
        else:
            # Choose action stochastically
            action = np.random.choice(np.arange(Game.ACTION_SIZE), p=pi)
        return action, pi

    def set_simulations(self, simulations: int):
        self._simulations = simulations

    def set_pi_turns(self, pi_turns: int):
        self._pi_turns = pi_turns


def build_player(name: str, player_class: type, **kwargs) -> IPlayer:
    """ Create a player
    :param name: player name
    :param player_class: IPlayer class
    :param kwargs: special arguments (if any) for player building
    """
    if player_class == Human:
        player = Human(name)
    elif player_class == ClassicMctsAgent:
        player = ClassicMctsAgent(name,kwargs['simulations'], kwargs['pi_turns'], kwargs['verbose'])
    elif player_class == MctsAgent:
        nnet = NeuralNet.create()
        load_folder = str(Path(ARCHIVE_DIR, GAME, '{:04}'.format(kwargs['dir_number'])))
        load_name = 'model{:04}.h5'.format(kwargs['model_version'])
        nnet.load(load_folder, load_name)
        player = MctsAgent(name, nnet, kwargs['simulations'], kwargs['pi_turns'], kwargs['verbose'])
    else:
        player = None
    return player
