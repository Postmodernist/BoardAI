import random
import sys
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from keras import regularizers
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, LeakyReLU, add
from keras.models import load_model, Model
from keras.optimizers import SGD

import config
import log
import paths
from game import Game, State
from loss import softmax_cross_entropy_with_logits


class ResidualCnn:
    """ Neural net that makes predictions about the value and actions probabilities for a given game state """

    def __init__(self, reg_const, learning_rate, momentum, input_dim, output_dim, hidden_layers):
        self.reg_const = reg_const
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.model = self._build_model()
        self.train_overall_loss = []
        self.train_value_loss = []
        self.train_policy_loss = []

    def predict(self, state: State):
        """ Predict state value and actions probability distribution
        :rtype state_value: float
        :rtype actions_prob_dist: np.ndarray
        """
        inputs = np.array([self._state_to_model_input(state)])
        value_out, policy_out = self.model.predict(inputs)
        state_value = value_out[0][0]
        logits = policy_out[0]
        mask = np.ones(logits.shape, dtype=bool)
        mask[state.allowed_actions] = False
        logits[mask] = -100  # mask invalid actions
        odds = np.exp(logits)
        actions_prob_dist = odds / np.sum(odds)  # normalize
        return state_value, actions_prob_dist

    def retrain(self, memory: deque):
        """ Retrain model """
        for i in range(config.TRAINING_LOOPS):
            mini_batch = random.sample(memory, min(config.BATCH_SIZE, len(memory)))
            training_states = np.array([self._state_to_model_input(item['state']) for item in mini_batch])
            training_targets = {
                'value_head': np.array([item['value'] for item in mini_batch]),
                'policy_head': np.array([item['actions_prob_dist'] for item in mini_batch])}
            # Train model
            fit = self.model.fit(training_states, training_targets, epochs=config.EPOCHS, verbose=1,
                                 validation_split=0, batch_size=32)
            log.model.info('Loss = {:.6f}, Policy loss = {:.6f}, Value loss = {:.6f}'.format(
                fit.history['loss'][0], fit.history['policy_head_loss'][0], fit.history['value_head_loss'][0]))
            self.train_overall_loss.append(round(fit.history['loss'][config.EPOCHS - 1], 4))
            self.train_value_loss.append(round(fit.history['value_head_loss'][config.EPOCHS - 1], 4))
            self.train_policy_loss.append(round(fit.history['policy_head_loss'][config.EPOCHS - 1], 4))
        self._plot_train_losses()
        self._log_weight_averages()

    @staticmethod
    def create():
        return ResidualCnn(config.REG_CONST, config.LEARNING_RATE, config.MOMENTUM, Game.input_shape, Game.board_size,
                           config.HIDDEN_CNN_LAYERS)

    @staticmethod
    def read(path: str, verbose=True):
        """ Read model from file """
        if verbose:
            print('Loading model... ', end='')
            sys.stdout.flush()
        m = load_model(path, custom_objects={'softmax_cross_entropy_with_logits': softmax_cross_entropy_with_logits})
        if verbose:
            print('done')
        return m

    def write(self, path: str, verbose=True):
        """ Write model to file """
        if verbose:
            print('Saving model... ', end='')
            sys.stdout.flush()
        self.model.save(path)
        if verbose:
            print('done')

    def view_layers(self):
        """ Plot model layers"""
        layers = self.model.layers
        for i, layer in enumerate(layers):
            layer_weights = layer.get_weights()
            print('Layer ' + str(i))
            try:
                weights = layer_weights[0]
                fig = plt.figure(figsize=(weights.shape[2], weights.shape[3]))  # width, height in inches
                w_channel = 0
                w_filter = 0
                for j in range(weights.shape[2] * weights.shape[3]):
                    sub = fig.add_subplot(weights.shape[3], weights.shape[2], j + 1)
                    sub.imshow(weights[:, :, w_channel, w_filter], cmap='coolwarm', clim=(-1, 1), aspect="auto")
                    w_channel = (w_channel + 1) % weights.shape[2]
                    w_filter = (w_filter + 1) % weights.shape[3]
                plt.savefig('{}plots/layers.png'.format(paths.RUN))
            except IndexError:
                try:
                    fig = plt.figure(figsize=(3, len(layer_weights)))  # width, height in inches
                    for j in range(len(layer_weights)):
                        sub = fig.add_subplot(len(layer_weights), 1, j + 1)
                        sub.imshow([layer_weights[j]], cmap='coolwarm', clim=(0, 2), aspect="auto")
                    plt.savefig('{}plots/layers.png'.format(paths.RUN))
                except IndexError:
                    try:
                        fig = plt.figure(figsize=(3, 3))  # width, height in inches
                        sub = fig.add_subplot(1, 1, 1)
                        sub.imshow(layer_weights[0], cmap='coolwarm', clim=(-1, 1), aspect="auto")
                        plt.savefig('{}plots/layers.png'.format(paths.RUN))
                    except IndexError:
                        pass

    def _build_model(self):
        main_input = Input(shape=self.input_dim, name='main_input')
        x = self._conv_layer(main_input, self.hidden_layers[0]['filters'], self.hidden_layers[0]['kernel_size'])
        if len(self.hidden_layers) > 1:
            for h in self.hidden_layers[1:]:
                x = self._residual_layer(x, h['filters'], h['kernel_size'])
        vh = self._value_head(x)
        ph = self._policy_head(x)
        model = Model(inputs=[main_input], outputs=[vh, ph])
        model.compile(loss={'value_head': 'mean_squared_error', 'policy_head': softmax_cross_entropy_with_logits},
                      optimizer=SGD(lr=self.learning_rate, momentum=self.momentum),
                      loss_weights={'value_head': 0.5, 'policy_head': 0.5})
        return model

    def _conv_layer(self, x, filters, kernel_size):
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   padding='same',
                   data_format="channels_first",
                   activation='linear',
                   use_bias=False,
                   kernel_regularizer=regularizers.l2(self.reg_const))(x)
        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)
        return x

    def _residual_layer(self, input_block, filters, kernel_size):
        x = self._conv_layer(input_block, filters, kernel_size)
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   padding='same',
                   data_format="channels_first",
                   activation='linear',
                   use_bias=False,
                   kernel_regularizer=regularizers.l2(self.reg_const))(x)
        x = BatchNormalization(axis=1)(x)
        x = add([input_block, x])
        x = LeakyReLU()(x)
        return x

    def _value_head(self, x):
        x = Conv2D(filters=1,
                   kernel_size=(1, 1),
                   padding='same',
                   data_format="channels_first",
                   activation='linear',
                   use_bias=False,
                   kernel_regularizer=regularizers.l2(self.reg_const))(x)
        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        x = Dense(units=20,
                  activation='linear',
                  use_bias=False,
                  kernel_regularizer=regularizers.l2(self.reg_const))(x)
        x = LeakyReLU()(x)
        x = Dense(units=1,
                  activation='tanh',
                  use_bias=False,
                  kernel_regularizer=regularizers.l2(self.reg_const),
                  name='value_head')(x)
        return x

    def _policy_head(self, x):
        x = Conv2D(filters=2,
                   kernel_size=(1, 1),
                   padding='same',
                   data_format="channels_first",
                   activation='linear',
                   use_bias=False,
                   kernel_regularizer=regularizers.l2(self.reg_const))(x)
        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        x = Dense(units=self.output_dim,
                  use_bias=False,
                  activation='linear',
                  kernel_regularizer=regularizers.l2(self.reg_const),
                  name='policy_head')(x)
        return x

    def _plot_train_losses(self):
        """ Plot training losses """
        plt.plot(self.train_overall_loss, 'k', linewidth=1.0)
        plt.plot(self.train_value_loss, 'g', linewidth=1.0)
        plt.plot(self.train_policy_loss, 'b', linewidth=1.0)
        plt.legend(['Loss', 'Value loss', 'Policy loss'], loc='lower left')
        plt.savefig('{}plots/train_losses.png'.format(paths.RUN))

    def _log_weight_averages(self):
        """ Write weights averages to log """
        layers = self.model.layers
        for i, layer in enumerate(layers):
            w = layer.get_weights()
            if len(w) < 1:
                continue
            w = w[0]
            log.model.info('Layer {:2}: mean_abs = {:.6f}, std = {:.6f}, max_abs = {:.6f}, min_abs = {:.6f}'
                           .format(i, np.mean(np.abs(w)), np.std(w), np.max(np.abs(w)), np.min(np.abs(w))))
        for i, layer in enumerate(layers):
            w = layer.get_weights()
            if len(w) < 2:
                continue
            w = w[1]
            log.model.info('Bias {:2}: mean_abs = {:.6f}, std = {:.6f}, max_abs = {:.6f}, min_abs = {:.6f}'
                           .format(i, np.mean(np.abs(w)), np.std(w), np.max(np.abs(w)), np.min(np.abs(w))))
        log.model.info('')

    def _state_to_model_input(self, state: State):
        return state.binary.reshape(self.input_dim)
