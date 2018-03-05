import numpy as np

from config import N

BOARD_SHAPE = (N, N)
BOARD_SIZE = N ** 2
ZERO_BOARD = np.zeros(BOARD_SIZE, dtype=np.int)
INDEX_BOARD = np.arange(BOARD_SIZE)
PIECES = {1: 'A', -1: 'B', 0: '.'}


def is_border(square: int) -> bool:
    """
    :return: True if square is on the border of the board
    """
    rem = square % N
    return square < N or square >= BOARD_SIZE - N or rem == 0 or rem == N - 1


BORDER_SQUARES = set(filter(is_border, INDEX_BOARD))


def get_neighbors() -> dict:
    """
     :return: dict of neighbors of each square
     """
    neighbors = {}
    for i in INDEX_BOARD:
        if i == 0:  # upper left corner
            neighbors[i] = [i + 1, i + N, i + N + 1]
        elif i == N - 1:  # upper right corner
            neighbors[i] = [i - 1, i + N - 1, i + N]
        elif i == N * (N - 1):  # lower left corner
            neighbors[i] = [i - N, i - N + 1, i + 1]
        elif i == N * N - 1:  # lower right corner
            neighbors[i] = [i - N - 1, i - N, i - 1]
        elif i < N:  # upper row
            neighbors[i] = [i - 1, i + 1, i + N - 1, i + N, i + N + 1]
        elif i > N * (N - 1):  # lower row
            neighbors[i] = [i - N - 1, i - N, i - N + 1, i - 1, i + 1]
        elif i % N == 0:  # left column
            neighbors[i] = [i - N, i - N + 1, i + 1, i + N, i + N + 1]
        elif i % N == N - 1:  # right column
            neighbors[i] = [i - N - 1, i - N, i - 1, i + N - 1, i + N]
        else:  # inner squares
            neighbors[i] = [i - N - 1, i - N, i - N + 1, i - 1, i + 1, i + N - 1, i + N, i + N + 1]
    return neighbors


NEIGHBORS = get_neighbors()


def update_valid_actions(board: np.ndarray, valid_actions: set, action: int) -> set:
    """
    :return: a list of valid actions
    """
    if valid_actions is None:
        return BORDER_SQUARES.copy()
    valid_actions.remove(action)
    new_actions = []
    for a in NEIGHBORS[action]:
        if board[a] == 0:
            new_actions.append(a)
    valid_actions.update(new_actions)
    return valid_actions


def get_win_segments() -> dict:
    """
    :return: dict of all possible win segments of each square of the board """

    def get_segments(i):
        row = [[i + k + l for k in range(4)] for l in range(-3, 1)]
        column = [[i + N * (k + l) for k in range(4)] for l in range(-3, 1)]
        diagonal1 = [[i + (N + 1) * (k + l) for k in range(4)] for l in range(-3, 1)]
        diagonal2 = [[i + (N - 1) * (k + l) for k in range(4)] for l in range(-3, 1)]
        return row + column + diagonal1 + diagonal2

    def is_valid(segment):
        # Out of bounds
        for i in segment:
            if i < 0 or i >= BOARD_SIZE:
                return False
        # Tearing
        for i in range(3):
            if abs(segment[i] % N - segment[i + 1] % N) > 1:
                return False
        return True

    win_segments = {}
    for square in INDEX_BOARD:
        win_segments[square] = list(filter(is_valid, get_segments(square)))
    return win_segments


WIN_SEGMENTS = get_win_segments()


def is_player_won(board: np.ndarray, player: int, action: int) -> bool:
    """
    :return: True if player has 4 pieces in a row
    """
    for segment in WIN_SEGMENTS[action]:
        failed = False
        for i in segment:
            if board[i] != player:
                failed = True
                break
        if not failed:
            return True
    return False


def get_symmetries(board: np.ndarray, pi: np.ndarray) -> list:
    """
    :param board: reshaped board
    :param pi: probability distribution vector
    :return: 8 symmetries of board and pi
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
