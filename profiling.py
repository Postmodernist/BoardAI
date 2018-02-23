import cProfile
import sys

import config
import log
from agent import Agent
from main import load_model, batch_play

EPISODES = 5

# Load model
current_nn, best_nn, best_player_version, memory = load_model()

# Create players
print('Creating players... ', end='')
sys.stdout.flush()
best_player = Agent('Best_player', best_nn)
print('done')

# Test run without profiling
if False:
    batch_play(EPISODES, best_player, best_player, config.TAU, memory, log.main)

# Run with profiling
cProfile.run('batch_play(EPISODES, best_player, best_player, config.TAU, memory, log.main)', sort='tottime')
