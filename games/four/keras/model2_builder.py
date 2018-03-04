from keras.layers import *
from keras.models import *
from keras.optimizers import *

from config import KERAS2_FILTERS as FILTERS, KERAS2_KERNEL_SIZE as KERNEL_SIZE, \
    KERAS2_LEARNING_RATE as LEARNING_RATE, KERAS2_MOMENTUM as MOMENTUM, \
    KERAS2_REG_CONST as REG_CONST, KERAS2_RESIDUAL_LAYERS as RESIDUAL_LAYERS


def build_model2(board_shape: tuple, action_size: int):
    board_x, board_y = board_shape
    input_boards = Input(shape=board_shape)
    x_image = Reshape((board_x, board_y, 1))(input_boards)
    x = conv_layer(x_image, FILTERS, KERNEL_SIZE)
    for _ in range(RESIDUAL_LAYERS):
        x = residual_layer(x)
    pi = policy_head(x, action_size)
    v = value_head(x)
    model = Model(inputs=input_boards, outputs=[pi, v])
    model.compile(loss={'pi': 'categorical_crossentropy', 'v': 'mean_squared_error'},
                  optimizer=SGD(LEARNING_RATE, MOMENTUM))
    return model


def conv_layer(x, filters, kernel_size):
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               padding='same',
               activation='linear',
               use_bias=False,
               kernel_regularizer=regularizers.l2(REG_CONST))(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU()(x)
    return x


def residual_layer(input_block):
    x = conv_layer(input_block, FILTERS, KERNEL_SIZE)
    x = Conv2D(filters=FILTERS,
               kernel_size=KERNEL_SIZE,
               padding='same',
               activation='linear',
               use_bias=False,
               kernel_regularizer=regularizers.l2(REG_CONST))(x)
    x = BatchNormalization(axis=3)(x)
    x = add([input_block, x])
    x = LeakyReLU()(x)
    return x


def policy_head(x, action_size):
    x = conv_layer(x, 2, 1)
    x = Flatten()(x)
    x = Dense(units=action_size,
              activation='linear',
              use_bias=False,
              kernel_regularizer=regularizers.l2(REG_CONST),
              name='pi')(x)
    return x


def value_head(x):
    x = conv_layer(x, 1, 1)
    x = Flatten()(x)
    x = Dense(units=20,
              activation='linear',
              use_bias=False,
              kernel_regularizer=regularizers.l2(REG_CONST))(x)
    x = LeakyReLU()(x)
    x = Dense(units=1,
              activation='tanh',
              use_bias=False,
              kernel_regularizer=regularizers.l2(REG_CONST),
              name='v')(x)
    return x
