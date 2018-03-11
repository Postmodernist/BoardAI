# Game
GAME = 'four'
N = 7  # length of the board side

# Memory and model loading parameters (see utils/paths.py for format description)
LOAD_DIR_NUMBER = None
MEMORY_VERSION = None
MODEL_VERSION = None

# Neural Net
MODEL = 'keras'
EPOCHS = 10
BATCH_SIZE = 64
CUDA = True
# 'keras' model
KERAS_LEARNING_RATE = 0.001
KERAS_DROPOUT = 0.3
KERAS_FILTERS = 512
KERAS_KERNEL_SIZE = 3
# 'keras2' model
KERAS2_LEARNING_RATE = 0.01
KERAS2_MOMENTUM = 0.9
KERAS2_REG_CONST = 0.0001
KERAS2_RESIDUAL_LAYERS = 6
KERAS2_FILTERS = 256
KERAS2_KERNEL_SIZE = 3

# Training
SELF_PLAY_EPISODES = 200
STOCHASTIC_TURNS = 15
MCTS_TRAIN_SIMULATIONS = 1000
MCTS_COMPETITIVE_SIMULATIONS = 1000
C_PUCT = 1.0
ALPHA = 0.8  # dirichlet noise parameter
EPSILON = 0.2  # noise fraction
EVAL_EPISODES = 40
EVAL_THRESHOLD = 0.6
MEMORY_SIZE = 200000

# Classic MCTS
CLASSIC_C_UCT = 1.41

# Logging
LOG_DISABLED = {'train': False, 'eval': False, 'mcts': True}
