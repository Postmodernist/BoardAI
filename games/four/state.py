import numpy as np

from intefraces.i_game_state import IGameState
from .logic import BOARD_SHAPE, PIECES, get_valid_actions, is_player_won


class State(IGameState):

    def __init__(self, board: np.ndarray, player: int, turn: int = 0):
        self._board = board
        self._player = player
        self._turn = turn
        self._valid_actions = get_valid_actions(board)
        self._opponent_won = is_player_won(board, -player)
        self._finished = not any(self._valid_actions) or self._opponent_won
        self._value = -1 if self._opponent_won else 0

    def get_board(self) -> np.ndarray:
        return self._board

    def get_canonical_board(self) -> np.ndarray:
        return np.reshape(self._board * self._player, BOARD_SHAPE)

    def get_player(self) -> int:
        return self._player

    def get_turn(self) -> int:
        return self._turn

    def get_valid_actions(self) -> np.ndarray:
        return self._valid_actions

    def get_next_state(self, action: int) -> IGameState:
        next_board = self._board.copy()
        next_board[action] = self._player
        return State(next_board, -self._player, self._turn + 1)

    def is_game_finished(self) -> bool:
        return self._finished

    def get_value(self) -> int:
        return self._value

    def get_hashable(self) -> bytes:
        return self._board.tostring()

    def log(self, logger):
        if logger.disabled:
            return
        for i, row in enumerate(self._board.reshape(BOARD_SHAPE)):
            logger.info('{:3} {}'.format(i * BOARD_SHAPE[0], ' '.join(PIECES[x] for x in row)))
