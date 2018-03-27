import os
from pathlib import Path
from sys import stdout

import tensorflow as tf
from keras import backend as K
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib

from config import GAME
from utils.loaders import NeuralNet
from utils.paths import TEMP_DIR, ARCHIVE_DIR

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # suppress tf messages

MODEL_NAME = 'model'
DIR_NUMBER = 21
MODEL_VERSION = 7


def load_model(dir_number, model_version):
    K.set_learning_phase(0)
    nnet = NeuralNet.create()
    load_folder = str(Path(ARCHIVE_DIR, GAME, '{:04}'.format(dir_number)))
    load_name = 'model{:04}.h5'.format(model_version)
    nnet.load(load_folder, load_name)


def export_model(saver: tf.train.Saver, input_node_names, output_node_names):
    print('Saving graph... ')
    stdout.flush()

    tf.train.write_graph(
        graph_or_graph_def=K.get_session().graph_def,
        logdir=TEMP_DIR,
        name=MODEL_NAME + '_graph.pbtxt')

    saver.save(
        sess=K.get_session(),
        save_path=str(Path(TEMP_DIR, MODEL_NAME + '.chkp')))

    freeze_graph.freeze_graph(
        input_graph=str(Path(TEMP_DIR, MODEL_NAME + '_graph.pbtxt')),
        input_saver=None,
        input_binary=False,
        input_checkpoint=str(Path(TEMP_DIR, MODEL_NAME + '.chkp')),
        output_node_names=output_node_names,
        restore_op_name='save/restore_all',
        filename_tensor_name='save/Const:0',
        output_graph=str(Path(TEMP_DIR, 'frozen_' + MODEL_NAME + '.pb')),
        clear_devices=True,
        initializer_nodes="")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open(str(Path(TEMP_DIR, 'frozen_' + MODEL_NAME + '.pb')), 'rb') as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def=input_graph_def,
        input_node_names=[input_node_names],
        output_node_names=output_node_names.split(','),
        placeholder_type_enum=tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile(str(Path(TEMP_DIR, 'opt_' + MODEL_NAME + '.pb')), 'wb') as f:
        f.write(output_graph_def.SerializeToString())

    print('Done')


# Load model
load_model(
    dir_number=DIR_NUMBER,
    model_version=MODEL_VERSION)

# To get input and output nodes names set a breakpoint after the model is created
# and look for names in the environment/model
export_model(
    saver=tf.train.Saver(),
    input_node_names='input_1',
    output_node_names='pi/Softmax,v/Tanh')
