from importlib import import_module

from config import LOAD_DIR_NUMBER, MEMORY_VERSION, MODEL_VERSION, MEMORY_SIZE
from intefraces.i_game import IGame
from intefraces.i_neural_net import INeuralNet
from memory import Memory
from utils.paths import GAME_MODULE_PATH, NNET_MODULE_PATH, MEMORY_FOLDER, MEMORY_NAME, MODEL_FOLDER, MODEL_NAME


def load_game_class() -> IGame:
    return getattr(import_module(GAME_MODULE_PATH), 'Game')


def load_nnet_class() -> INeuralNet:
    return getattr(import_module(NNET_MODULE_PATH), 'NeuralNet')


Game = load_game_class()
NeuralNet = load_nnet_class()


def load_memory() -> Memory:
    """
    :return memory: new or loaded memory object
    """
    if LOAD_DIR_NUMBER is None or MEMORY_VERSION is None:
        return Memory.create(MEMORY_SIZE)
    memory = Memory(MEMORY_SIZE)
    memory.load(MEMORY_FOLDER, MEMORY_NAME)
    return memory


def load_model() -> (INeuralNet, INeuralNet, int):
    """
    :returns
        best_nn: best NeuralNet object
        adversary_nn: adversary NeuralNet object
        best_model_version: current version of the best NeuralNet
    """
    best_nn = NeuralNet.create()
    if LOAD_DIR_NUMBER is None or MODEL_VERSION is None:
        best_model_version = 0
    else:
        best_model_version = MODEL_VERSION
        best_nn.load(MODEL_FOLDER, MODEL_NAME)
    adversary_nn = NeuralNet.create()
    adversary_nn.model.set_weights(best_nn.model.get_weights())
    return best_nn, adversary_nn, best_model_version
