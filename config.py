# Self play
EPISODES = 30
MCTS_SIMULATIONS = 50
MEMORY_SIZE = 30000
STOCHASTIC_TURNS = 10
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
