import matplotlib.pyplot as plt
import numpy as np
from keras import regularizers
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, LeakyReLU, add
from keras.models import load_model, Model
from keras.optimizers import SGD

import config
from loggers import log_model
from loss import softmax_cross_entropy_with_logits


class ResidualCnn:

    def __init__(self, reg_const, learning_rate, input_dim, output_dim, hidden_layers):
        self.reg_const = reg_const
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.num_layers = len(hidden_layers)
        self.model = self._build_model()

    def predict(self, x):
        """ Make a prediction """
        return self.model.predict(x)

    def fit(self, states, targets, epochs, verbose, validation_split, batch_size):
        """ Fit model """
        return self.model.fit(states, targets, epochs=epochs, verbose=verbose, validation_split=validation_split,
                              batch_size=batch_size)

    def write(self, version):
        """ Write model to file """
        filename = 'models/version{0:0>4}.h5'.format(version)
        self.model.save(config.RUN_PATH + filename)

    @staticmethod
    def read(game, run_number, version):
        """ Read model from file """
        filename = 'models/version{0:0>4}.h5'.format(version)
        path = config.RUN_ARCHIVE_PATH + game + '/run{0:0>4}/'.format(run_number) + filename
        return load_model(path, custom_objects={'softmax_cross_entropy_with_logits': softmax_cross_entropy_with_logits})

    def print_weight_averages(self):
        """ Log layers stats """
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

    def state_to_model_input(self, state):
        return state.binary.reshape(self.input_dim)

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
                      optimizer=SGD(lr=self.learning_rate, momentum=config.MOMENTUM),
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
