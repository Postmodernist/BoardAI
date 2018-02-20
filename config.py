# Self play
STOP_TRAINING = False  # change to True and save to stop training loop
EPISODES = 30  # number of self-play games
MCTS_SIMULATIONS = 50  # number of MCTS simulations
MEMORY_SIZE = 30000  # memory size threshold to trigger neural net retraining
STOCHASTIC_TURNS = 10  # number of stochastic turns before switching to deterministic
C = 1  # coefficient of second term of probabilistic UCT
EPSILON = 0.2  # exploration noise influence
ALPHA = 0.8  # dirichlet noise generation parameter

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
SCORING_THRESHOLD = 1.3  # current player must be this much better than the best player

# Misc
PLOT_MODEL_GRAPH = False
