# Initialize
INITIAL_RUN_NUMBER = None
INITIAL_MEMORY_VERSION = None
INITIAL_MODEL_VERSION = None

# Paths
RUN_PATH = './run/'
RUN_ARCHIVE_PATH = './run_archive/'

# Self play
EPISODES = 30
MCTS_SIMS = 50
MEMORY_SIZE = 30000
TURNS_UNTIL_TAU0 = 10  # on this turn start playing deterministically
CPUCT = 1
EPSILON = 0.2
ALPHA = 0.8

# Retraining
BATCH_SIZE = 256
EPOCHS = 1
REG_CONST = 0.0001
LEARNING_RATE = 0.1
MOMENTUM = 0.9
TRAINING_LOOPS = 10
HIDDEN_CNN_LAYERS = [
    {'filters': 75, 'kernel_size': (4, 4)},
    {'filters': 75, 'kernel_size': (4, 4)},
    {'filters': 75, 'kernel_size': (4, 4)},
    {'filters': 75, 'kernel_size': (4, 4)},
    {'filters': 75, 'kernel_size': (4, 4)},
    {'filters': 75, 'kernel_size': (4, 4)}]

# Evaluation
EVAL_EPISODES = 20
SCORING_THRESHOLD = 1.3
