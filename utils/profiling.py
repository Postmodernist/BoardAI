import cProfile

from evaluate import Evaluate
from players import ClassicMctsAgent, build_player
from utils.loggers import null as null_log

episodes = 2
player = build_player('Botik', ClassicMctsAgent, pi_turns=0, verbose=False)
play = Evaluate(player, player, null_log, verbose=True)
cProfile.run('play.batch(episodes, "Profiling run")', sort='tottime')
