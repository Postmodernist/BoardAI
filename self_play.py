import time

from intefraces.i_player import IPlayer
from memory import Memory
from utils.average_meter import AverageMeter
from utils.loaders import Game
from utils.loggers import training as train_log
from utils.progress import progress_bar, sec_to_time


class SelfPlay:

    def __init__(self, player: IPlayer, logger=train_log, verbose=True):
        self._player = player
        self._logger = logger
        self._verbose = verbose

    def single(self, memory: Memory):
        """ Play a single match """
        state = Game.get_initial_state()
        state.log(self._logger)
        turn = 0
        while not state.is_game_finished():
            turn += 1
            self._logger.info('Turn {}'.format(turn))
            # Get action
            action, pi = self._player.get_action(state)
            # Commit example
            symmetries = Game.get_symmetries(state.get_canonical_board(), pi)
            for sym in symmetries:
                memory.append(sym + (state.get_player(), None))
            # Make the move
            state = state.get_next_state(action)
            state.log(self._logger)
        # Update examples
        memory.short = list(
            map(lambda x: (x[0], x[1], state.get_value() if state.get_player() == x[2] else -state.get_value()),
                memory.short))
        memory.commit()

    def batch(self, episodes: int, memory: Memory, prefix: str = ''):
        """ Play a batch of matches
        :param episodes: number of matches to play
        :param memory: Memory object to store examples
        :param prefix: progress bar prefix
        """
        episode_time = AverageMeter()
        if self._verbose:
            progress_bar(0, episodes, prefix, 'Playing first episode...')
        t_start = time.time()
        t = time.time()
        for e in range(episodes):
            self._logger.info('************ Episode {} of {} ************'.format(e, episodes))
            # Play a match
            self.single(memory)
            # Output progress
            if self._verbose:
                episode_time.update(time.time() - t)
                t = time.time()
                eta = t_start + episode_time.avg * episodes - t
                suffix = 'Episode time: {:.3f}s | Total: {} | ETA: {}'.format(
                    episode_time.avg, sec_to_time(round(episode_time.sum)), sec_to_time(round(eta)))
                progress_bar(e, episodes, prefix, suffix)
