# Self play
STOP_TRAINING = False  # change to True and save to stop training loop
EPISODES = 30  # number of self-play games
MCTS_SIMULATIONS = 200  # number of MCTS simulations
MEMORY_SIZE = 30000  # memory size threshold to trigger neural net retraining
TAU = 10  # number of turns before switching to deterministic play
C = 1.4  # coefficient of second term of probabilistic UCT
EPSILON = 0.2  # exploration noise influence
ALPHA = 0.8  # dirichlet noise generation parameter

# Retraining
BATCH_SIZE = 512
EPOCHS = 1
REG_CONST = 0.0001
LEARNING_RATE = 0.1
MOMENTUM = 0.9
TRAINING_LOOPS = 10
HIDDEN_CNN_LAYERS = [
    {'filters': 128, 'kernel_size': (3, 3)},
    {'filters': 128, 'kernel_size': (3, 3)},
    {'filters': 128, 'kernel_size': (3, 3)},
    {'filters': 128, 'kernel_size': (3, 3)},
    {'filters': 128, 'kernel_size': (3, 3)},
    {'filters': 128, 'kernel_size': (3, 3)},
    {'filters': 128, 'kernel_size': (3, 3)},
    {'filters': 128, 'kernel_size': (3, 3)},
    {'filters': 128, 'kernel_size': (3, 3)},
    {'filters': 128, 'kernel_size': (3, 3)}]

# Evaluation
EVAL_EPISODES = 20
WINS_RATIO = 0.55  # current player must win this percent of games to be declared the new best

# Misc
PLOT_MODEL_GRAPH = False
ALTERNATIVE_UCT = False  # use alternative version of probabilistic UCT function
