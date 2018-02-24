# Self play
EPISODES = 50  # number of self-play games (AZ: 25000)
TAU = 15  # number of turns before switching to deterministic play

# Evaluation
EVAL_EPISODES = 30  # number of tournament games (AZ: 400)
WINS_RATIO = 0.55  # current player must win this percent of games to be declared the new best (AZ: 0.55)

# MCTS
MCTS_SIMULATIONS = 200  # number of MCTS simulations per turn (AZ: 1600)
C = 1.4  # coefficient of U
EPSILON = 0.2  # noise influence
ALPHA = 0.8  # dirichlet noise generation parameter

# Model retraining
MEMORY_SIZE = 50000  # number of evaluated positions from last games (AZ: ~100,000,000)
BATCH_SIZE = 512  # mini-batch size (AZ: 2048)
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
    {'filters': 128, 'kernel_size': (3, 3)}]  # (AZ: 40 layers, 256 filters, 3x3 kernel)
