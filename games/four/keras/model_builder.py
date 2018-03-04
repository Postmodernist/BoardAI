from keras.layers import *
from keras.models import *
from keras.optimizers import *

from config import KERAS_FILTERS as FILTERS, KERAS_KERNEL_SIZE as KERNEL_SIZE, \
    KERAS_LEARNING_RATE as LEARNING_RATE, KERAS_DROPOUT as DROPOUT


def build_model(board_shape: tuple, action_size: int):
    board_x, board_y = board_shape
    # s: BATCH_SIZE x board_x x board_y
    input_boards = Input(shape=board_shape)
    # BATCH_SIZE x board_x x board_y x 1
    x_image = Reshape((board_x, board_y, 1))(input_boards)
    # BATCH_SIZE x board_x x board_y x FILTERS
    h_conv1 = conv_layer(x_image, padding='same')
    # BATCH_SIZE x board_x x board_y x FILTERS
    h_conv2 = conv_layer(h_conv1, padding='same')
    # BATCH_SIZE x (board_x - 2) x (board_y - 2) x FILTERS
    h_conv3 = conv_layer(h_conv2, padding='valid')
    # BATCH_SIZE x (board_x - 4) x (board_y - 4) x FILTERS
    h_conv4 = conv_layer(h_conv3, padding='valid')
    h_conv4_flat = Flatten()(h_conv4)
    # BATCH_SIZE x 1024
    s_fc1 = dropout_layer(h_conv4_flat, units=1024)
    # BATCH_SIZE x 512
    s_fc2 = dropout_layer(s_fc1, units=512)
    # BATCH_SIZE x ACTION_SIZE
    pi = Dense(action_size, activation='softmax', name='pi')(s_fc2)
    # BATCH_SIZE x 1
    v = Dense(1, activation='tanh', name='v')(s_fc2)
    model = Model(inputs=input_boards, outputs=[pi, v])
    model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(LEARNING_RATE))
    return model


def conv_layer(x, padding):
    x = Conv2D(FILTERS, KERNEL_SIZE, padding=padding)(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    return x


def dropout_layer(x, units):
    x = Dense(units=units)(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = Dropout(DROPOUT)(x)
    return x
