from pathlib import Path
from sys import stdout

import matplotlib.pyplot as plt
import numpy as np

from config import BATCH_SIZE, EPOCHS
from intefraces.i_neural_net import INeuralNet
from utils.loaders import Game
from utils.paths import PLOT_LOSSES_FOLDER, PLOT_LOSSES_NAME
from .model2_builder import build_model2
from .model_builder import build_model


class NeuralNet(INeuralNet):

    def __init__(self):
        if True:  # choose model
            self.model = build_model(Game.BOARD_SHAPE, Game.ACTION_SIZE)
        else:
            self.model = build_model2(Game.BOARD_SHAPE, Game.ACTION_SIZE)
        self._loss = []
        self._value_loss = []
        self._policy_loss = []

    @staticmethod
    def create() -> INeuralNet:
        print('Creating uninitialized model... ', end='')
        stdout.flush()
        nnet = NeuralNet()
        print('done')
        return nnet

    def train(self, examples: list):
        """
        :param examples: a list of training examples, each example of form (board, pi, v)
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        fit = self.model.fit(x=input_boards, y=[target_pis, target_vs], batch_size=BATCH_SIZE, epochs=EPOCHS)
        self._loss.append(round(fit.history['loss'][EPOCHS - 1], 4))
        self._value_loss.append(round(fit.history['v'][EPOCHS - 1], 4))
        self._policy_loss.append(round(fit.history['pi'][EPOCHS - 1], 4))
        self._plot_train_losses()

    def predict(self, canonical_board: np.ndarray, valid_actions: list):
        """
        :param canonical_board: current board in its canonical form
        :param valid_actions: a list of valid actions
        """
        canonical_board = canonical_board[np.newaxis, :, :]
        pi, v = self.model.predict(canonical_board)
        pi = pi[0]
        valid_actions_mask = np.zeros(Game.ACTION_SIZE, dtype=np.int)
        valid_actions_mask[valid_actions] = 1
        pi *= valid_actions_mask  # mask invalid actions
        pi /= sum(pi)  # normalize
        return pi, v[0][0]

    def save(self, folder: str, name: str):
        print('Saving model weights to {} ... '.format(name), end='')
        stdout.flush()
        Path(folder).mkdir(parents=True, exist_ok=True)
        path = Path(folder, name)
        self.model.save_weights(str(path))
        print('done')

    def load(self, folder: str, name: str):
        print('Loading model weights from {} ... '.format(name), end='')
        stdout.flush()
        path = Path(folder, name)
        if not path.exists():
            print()
            raise "File not found: {}".format(path)
        self.model.load_weights(str(path))
        print('done')

    def _plot_train_losses(self):
        """ Plot training losses """
        Path(PLOT_LOSSES_FOLDER).mkdir(parents=True, exist_ok=True)
        path = Path(PLOT_LOSSES_FOLDER, PLOT_LOSSES_NAME)
        plt.plot(self._loss, 'k', linewidth=1.0)
        plt.plot(self._value_loss, 'g', linewidth=1.0)
        plt.plot(self._policy_loss, 'b', linewidth=1.0)
        plt.legend(['Overall loss', 'Value loss', 'Policy loss'], loc='lower left')
        plt.savefig(str(path))
