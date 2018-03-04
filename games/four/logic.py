import numpy as np

from config import N

BOARD_SHAPE = (N, N)
BOARD_SIZE = N ** 2
ZERO_BOARD = np.zeros(BOARD_SIZE, dtype=np.int)
INDEX_BOARD = np.arange(BOARD_SIZE)
PIECES = {1: 'A', -1: 'B', 0: '.'}


def get_win_positions():
    """ Get all possible win combinations for the board """

    def get_rows(board: np.ndarray):
        """ All rows, columns and diagonals of the board """
        board = board.reshape(BOARD_SHAPE)
        board_flip = np.fliplr(board)
        rows = [board[0]]
        cols = [board[:, 0]]
        diagonals1 = [board.diagonal(0)]
        diagonals2 = [board_flip.diagonal(0)]
        for i in range(1, N):
            rows.append(board[i])
            cols.append(board[:, i])
            if N - i >= 4:
                diagonals1.append(board.diagonal(i))
                diagonals1.append(board.diagonal(-i))
                diagonals2.append(board_flip.diagonal(i))
                diagonals2.append(board_flip.diagonal(-i))
        return rows + cols + diagonals1 + diagonals2

    win_positions = []
    for row in get_rows(INDEX_BOARD):
        win_positions.extend([row[j:j + 4] for j in range(len(row) - 3)])
    return win_positions


WIN_POSITIONS = get_win_positions()


def is_border(pos: int):
    """
    :return True if pos is the border cell
    """
    rem = pos % N
    return pos < N or pos >= BOARD_SIZE - N or rem == 0 or rem == N - 1


BORDER_POSITIONS = set(filter(is_border, INDEX_BOARD))


def get_neighbors():
    """
     :return neighbors: indexes of neighbors for each inner cell
     """
    neighbors = {}
    for i in filter(lambda x: x not in BORDER_POSITIONS, INDEX_BOARD):
        neighbors[i] = [i - 1, i + 1, i - N, i + N, i - N - 1, i - N + 1, i + N - 1, i + N + 1]
    return neighbors


NEIGHBORS = get_neighbors()


def get_valid_actions(board: np.ndarray):
    """
    :return a list of valid actions
    """

    def is_valid_action(action: int):
        """
        :return: True if action is valid
        """
        # Cell is not empty
        if board[action] != 0:
            return False
        # Border cell
        if action in BORDER_POSITIONS:
            return True
        # Cell has non-empty neighbor
        for x in NEIGHBORS[action]:
            if board[x]:
                return True
        return False

    return list(filter(is_valid_action, INDEX_BOARD))


def is_player_won(board: np.ndarray, player: int):
    """
    :return True if player has a winning combination
    """
    for win_position in WIN_POSITIONS:
        failed = False
        for i in win_position:
            if board[i] != player:
                failed = True
                break
        if not failed:
            return True
    return False


def get_symmetries(board: np.ndarray, pi: np.ndarray) -> list:
    """ Generate 8 symmetries of board and pi
    :param board: reshaped board
    :param pi: probability distribution vector
    """

    def make_symmetry(a, b):
        return a, b.ravel()

    symmetries = []
    pi = pi.reshape(BOARD_SHAPE)
    symmetries.append(make_symmetry(board, pi))
    board_m = np.fliplr(board)
    pi_m = np.fliplr(pi)
    symmetries.append(make_symmetry(board_m, pi_m))
    for _ in range(3):
        board = np.rot90(board)
        pi = np.rot90(pi)
        symmetries.append(make_symmetry(board, pi))
        board_m = np.rot90(board_m)
        pi_m = np.rot90(pi_m)
        symmetries.append(make_symmetry(board_m, pi_m))
    return symmetries
