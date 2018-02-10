import os
import numpy as np
from config import *
from loggers import log_main
from game import Game

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
np.set_printoptions(suppress=True)
env = Game()

print(env.name)
print(env.state.allowed_actions)
