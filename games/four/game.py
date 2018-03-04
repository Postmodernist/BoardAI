from intefraces.i_game import IGame
from intefraces.i_game_state import IGameState
from .logic import BOARD_SHAPE, BOARD_SIZE, ZERO_BOARD, get_symmetries
from .state import State


class Game(IGame):
    """
    'Four' game. First player to get 4 marks in a straight line wins. Mark can be placed only on the
    border or in a cell that has a non-empty neighbor.
    """

    BOARD_SHAPE = BOARD_SHAPE
    BOARD_SIZE = BOARD_SIZE
    ACTION_SIZE = BOARD_SIZE

    @staticmethod
    def get_initial_state() -> IGameState:
        return State(ZERO_BOARD, 1)

    @staticmethod
    def get_symmetries(board, pi) -> list:
        return get_symmetries(board, pi)
