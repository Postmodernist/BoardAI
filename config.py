# Self play
EPISODES = 150  # number of self-play games (AZ: 25000)
TAU = 2  # temperature, parameter controlling exploration

# Evaluation
EVAL_EPISODES = 20  # number of tournament games (AZ: 400)
WINS_RATIO = 0.55  # current player must win this percent of games to be declared the new best (AZ: 0.55)

# MCTS
MCTS_SIMULATIONS = 100  # number of MCTS simulations per turn (AZ: 1600)
C = 1.4  # coefficient of U

# Model retraining
MEMORY_SIZE = 100000  # number of evaluated positions from last games (AZ: ~100,000,000)
BATCH_SIZE = 1024  # mini-batch size (AZ: 2048)
EPOCHS = 3
REG_CONST = 0.0001
LEARNING_RATE = 0.1
MOMENTUM = 0.9
TRAINING_LOOPS = 20
HIDDEN_CNN_LAYERS = [  # (AZ: 40 layers, 256 filters, 3x3 kernel)
    {'filters': 128, 'kernel_size': (4, 4)},
    {'filters': 128, 'kernel_size': (4, 4)},
    {'filters': 128, 'kernel_size': (4, 4)},
    {'filters': 128, 'kernel_size': (4, 4)},
    {'filters': 128, 'kernel_size': (4, 4)},
    {'filters': 128, 'kernel_size': (4, 4)},
    {'filters': 128, 'kernel_size': (4, 4)},
    {'filters': 128, 'kernel_size': (4, 4)},
    {'filters': 128, 'kernel_size': (4, 4)},
    {'filters': 128, 'kernel_size': (4, 4)}]
