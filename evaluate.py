import time

from intefraces.i_game_state import IGameState
from intefraces.i_player import IPlayer
from utils.average_meter import AverageMeter
from utils.loaders import Game
from utils.loggers import evaluation as eval_log
from utils.progress import progress_bar


class Evaluate:

    def __init__(self, player1: IPlayer, player2: IPlayer, logger=eval_log, verbose=True):
        self._player1 = player1
        self._player2 = player2
        self._logger = logger
        self._verbose = verbose

    def single(self) -> IGameState:
        """ Play a single match
        :return state: final game state
        """
        state = Game.get_initial_state()
        state.log(self._logger)
        player = self._player1
        opponent = self._player2
        turn = 0
        while not state.is_game_finished():
            turn += 1
            self._logger.info('Turn {} | Player {} ({})'.format(turn, state.get_player(), player.get_name()))
            # Get action
            action, _ = player.get_action(state)
            # Make the move
            state = state.get_next_state(action)
            state.log(self._logger)
            # Switch player
            player, opponent = opponent, player
        if state.get_value() == -1:
            self._logger.info('Game over | Player {} ({}) won'.format(-state.get_player(), opponent.get_name()))
        else:
            self._logger.info('Game over | Draw')
        return state

    def batch(self, episodes: int, prefix: str = '') -> dict:
        """ Play a batch of matches
        :param episodes: number of matches to play
        :param prefix: progress bar prefix
        :return wins: {player name: win count}
        """

        def play_half_episodes(e):
            t = time.time()
            for _ in range(episodes // 2):
                e += 1
                self._logger.info('************ Episode {} of {} ************'.format(e, episodes))
                # Play a match
                state = self.single()
                # Update win counts
                if state.get_value() == -1:
                    winner = self._player1 if state.get_player() == -1 else self._player2
                    wins[winner.get_name()] += 1
                else:
                    wins['draw'] += 1
                # Output progress
                if self._verbose:
                    episode_time.update(time.time() - t)
                    t = time.time()
                    suffix = 'Episode time: {:.3f}s | Total: {:1.1f}s | ETA: {:1.1f}s'.format(
                        episode_time.avg, episode_time.sum, t_start + episode_time.avg * episodes - t)
                    progress_bar(e, episodes, prefix, suffix)

        wins = {self._player1.get_name(): 0, self._player2.get_name(): 0, 'draw': 0}
        episode_time = AverageMeter()
        if self._verbose:
            progress_bar(0, episodes, prefix, 'Playing first episode...')
        # First player first
        t_start = time.time()
        play_half_episodes(0)
        # Swap players
        self._player1, self._player2 = self._player2, self._player1
        play_half_episodes(episodes // 2)
        # Write results
        if self._verbose:
            name1 = self._player1.get_name()
            name2 = self._player2.get_name()
            print('{} / {} / draw -- {}/{}/{}'.format(name1, name2, wins[name1], wins[name2], wins['draw']))
        return wins
