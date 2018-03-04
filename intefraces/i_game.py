class IGame:
    """
    This class specifies the base Game class. To define your own game, subclass this class and
    IGameState class, and implement the functions below. This works when the game is two-player,
    adversarial and turn-based.

    Use 1 for player1 and -1 for player2.
    """

    BOARD_SHAPE = (None, None)  # shape of the board
    BOARD_SIZE = None  # number of board cells
    ACTION_SIZE = None  # number of all possible actions

    @staticmethod
    def get_initial_state():
        """
        :return initial_state: new default game state object
        """
        pass

    @staticmethod
    def get_symmetries(board, pi):
        """
        :param board: board
        :param pi: policy vector of size ACTION_SIZE
        :return symmetries: a list of [(board, pi)] where each tuple is a symmetrical form of
            the board and the corresponding pi vector. This is used when training the neural
            network from examples, so board must be reshaped to BOARD_SHAPE
        """
        pass
