from intefraces.i_game import IGame
from intefraces.i_game_state import IGameState
from .logic import BOARD_SHAPE, BOARD_SIZE, ZERO_BOARD, get_symmetries
from .state import State


class Game(IGame):
    """
    'Four' game.The first player to have 4 pieces in a straight line wins.
    A player can't place a piece on a square surrounded by empty squares.
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
