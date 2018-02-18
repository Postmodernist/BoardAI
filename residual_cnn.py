import time

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from IPython import display
from keras import regularizers
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, LeakyReLU, add
from keras.models import load_model, Model
from keras.optimizers import SGD

import config
import paths
from loggers import log_model
from loss import softmax_cross_entropy_with_logits as softmax_loss


class ResidualCnn:
    """ Neural net that makes predictions about the value and actions probabilities for a given game state """

    def __init__(self, reg_const, learning_rate, momentum, input_dim, output_dim, hidden_layers):
        self.reg_const = reg_const
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.num_layers = len(hidden_layers)
        self.model = self._build_model()
        self.train_overall_loss = []
        self.train_value_loss = []
        self.train_policy_loss = []
        self.val_overall_loss = []
        self.val_value_loss = []
        self.val_policy_loss = []

    def predict(self, state):
        """ Make a prediction """
        inputs = np.array([self.state_to_model_input(state)])
        return self.model.predict(inputs)

    def retrain(self, memory):
        """ Retrain model """
        log_model.info('Retraining model...')
        for i in range(config.TRAINING_LOOPS):
            mini_batch = np.random.choice(memory, min(config.BATCH_SIZE, len(memory)), replace=False)
            training_states = np.array([self.state_to_model_input(row['state'])] for row in mini_batch)
            training_targets = {
                'value_head': np.array([row['value'] for row in mini_batch]),
                'policy_head': np.array([row['action_values'] for row in mini_batch])}
            fit = self.model.fit(training_states, training_targets, epochs=config.EPOCHS, verbose=1, validation_split=0,
                                 batch_size=32)
            log_model.info('New loss %s', fit.history)
            self.train_overall_loss.append(round(fit.history['loss'][config.EPOCHS - 1], 4))
            self.train_value_loss.append(round(fit.history['value_head_loss'][config.EPOCHS - 1], 4))
            self.train_policy_loss.append(round(fit.history['policy_head_loss'][config.EPOCHS - 1], 4))
        # Plot losses
        plt.plot(self.train_overall_loss, 'k')
        plt.plot(self.train_value_loss, 'k:')
        plt.plot(self.train_policy_loss, 'k--')
        plt.legend(['train_overall_loss', 'train_value_loss', 'train_policy_loss'], loc='lower left')
        display.clear_output(wait=True)
        display.display(pl.gcf())
        pl.gcf().clear()
        time.sleep(1.0)
        # Log weights averages
        layers = self.model.layers
        for i, layer in enumerate(layers):
            try:
                x = layer.get_weights()[0]
                log_model.info('Weight layer %d: mean_abs = %f, std =%f, max_abs =%f, min_abs =%f',
                               i, np.mean(np.abs(x)), np.std(x), np.max(np.abs(x)), np.min(np.abs(x)))
            except IndexError:
                pass
        for i, layer in enumerate(layers):
            try:
                x = layer.get_weights()[1]
                log_model.info('Bias layer %d: mean_abs = %f, std =%f, max_abs =%f, min_abs =%f',
                               i, np.mean(np.abs(x)), np.std(x), np.max(np.abs(x)), np.min(np.abs(x)))
            except IndexError:
                pass
        log_model.info('-' * 80)

    def state_to_model_input(self, state):
        return state.binary.reshape(self.input_dim)

    @staticmethod
    def read(path, version):
        """ Read model from file """
        path = '{}models/version{0:0>4}.h5'.format(path, version)
        return load_model(path, custom_objects={'softmax_cross_entropy_with_logits': softmax_loss})

    def write(self, version):
        """ Write model to file """
        path = '{}models/version{0:0>4}.h5'.format(paths.RUN, version)
        self.model.save(path)

    def view_layers(self):
        """ Plot model weights"""
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
                plt.show()
            except IndexError:
                try:
                    fig = plt.figure(figsize=(3, len(layer_weights)))  # width, height in inches
                    for j in range(len(layer_weights)):
                        sub = fig.add_subplot(len(layer_weights), 1, j + 1)
                        sub.imshow([layer_weights[j]], cmap='coolwarm', clim=(0, 2), aspect="auto")
                    plt.show()
                except IndexError:
                    try:
                        fig = plt.figure(figsize=(3, 3))  # width, height in inches
                        sub = fig.add_subplot(1, 1, 1)
                        sub.imshow(layer_weights[0], cmap='coolwarm', clim=(-1, 1), aspect="auto")
                        plt.show()
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
        model.compile(loss={'value_head': 'mean_squared_error', 'policy_head': softmax_loss},
                      optimizer=SGD(lr=self.learning_rate, momentum=self.momentum),
                      loss_weights={'value_head': 0.5, 'policy_head': 0.5})
        return model

    def _conv_layer(self, x, filters, kernel_size):
        x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same', data_format="channels_first",
                   activation='linear', use_bias=False, kernel_regularizer=regularizers.l2(self.reg_const))(x)
        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)
        return x

    def _residual_layer(self, input_block, filters, kernel_size):
        x = self._conv_layer(input_block, filters, kernel_size)
        x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same', data_format="channels_first",
                   activation='linear', use_bias=False, kernel_regularizer=regularizers.l2(self.reg_const))(x)
        x = BatchNormalization(axis=1)(x)
        x = add([input_block, x])
        x = LeakyReLU()(x)
        return x

    def _value_head(self, x):
        x = Conv2D(filters=1, kernel_size=(1, 1), padding='same', data_format="channels_first", activation='linear',
                   use_bias=False, kernel_regularizer=regularizers.l2(self.reg_const))(x)
        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        x = Dense(20, activation='linear', use_bias=False, kernel_regularizer=regularizers.l2(self.reg_const))(x)
        x = LeakyReLU()(x)
        x = Dense(1, activation='tanh', use_bias=False, kernel_regularizer=regularizers.l2(self.reg_const),
                  name='value_head')(x)
        return x

    def _policy_head(self, x):
        x = Conv2D(filters=2, kernel_size=(1, 1), padding='same', data_format="channels_first", activation='linear',
                   use_bias=False, kernel_regularizer=regularizers.l2(self.reg_const))(x)
        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        x = Dense(self.output_dim, use_bias=False, activation='linear',
                  kernel_regularizer=regularizers.l2(self.reg_const), name='policy_head')(x)
        return x
