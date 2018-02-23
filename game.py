import numpy as np

# Board
BOARD_SIDE = 7
BOARD_SIZE = BOARD_SIDE ** 2
BOARD_SHAPE = (BOARD_SIDE, BOARD_SIDE)
ZERO_BOARDS = np.zeros(BOARD_SIZE * 3, dtype=np.int)  # 3 boards
ZERO_BOARD = np.zeros(BOARD_SIZE, dtype=np.int)
ONE_BOARD = np.ones(BOARD_SIZE, dtype=np.int)
INDEX_BOARD = np.arange(BOARD_SIZE)
PIECES = {1: 'A', -1: 'B', 0: '.'}


def get_win_positions():
    """ Get all possible win combinations for the board """

    def get_rows(board):
        """ All rows, columns and diagonals of the board """
        board = board.reshape(BOARD_SHAPE)
        board_flip = np.fliplr(board)
        rows = [board[0]]
        cols = [board[:, 0]]
        diagonals1 = [board.diagonal(0)]
        diagonals2 = [board_flip.diagonal(0)]
        for i in range(1, BOARD_SIDE):
            rows.append(board[i])
            cols.append(board[:, i])
            if BOARD_SIDE - i >= 4:
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


def is_border(pos):
    """ Return true if pos is the border cell """
    rem = pos % BOARD_SIDE
    return pos < BOARD_SIDE or pos >= BOARD_SIZE - BOARD_SIDE or rem == 0 or rem == BOARD_SIDE - 1


BORDER_POSITIONS = set(filter(is_border, INDEX_BOARD))


def get_neighbors():
    """ Get neighbors for each inner cell """
    neighbors = {}
    for i in filter(lambda x: x not in BORDER_POSITIONS, INDEX_BOARD):
        neighbors[i] = [i - 1, i + 1, i - BOARD_SIDE, i + BOARD_SIDE, i - BOARD_SIDE - 1, i - BOARD_SIDE + 1,
                        i + BOARD_SIDE - 1, i + BOARD_SIDE + 1]
    return neighbors


NEIGHBORS = get_neighbors()


class Game:
    """ Four-in-a-row game. First player to get 4 marks in a straight line wins. Mark can be placed only on the
    border or in a cell that has a non-empty neighbor """

    name = 'four_in_a_row'
    board_size = BOARD_SIZE
    board_shape = BOARD_SHAPE
    input_shape = (7,) + BOARD_SHAPE
    pieces = PIECES

    def __init__(self):
        self.state = State(ZERO_BOARDS, 1)

    def reset(self):
        """ Reset game """
        self.state = State(ZERO_BOARDS, 1)
        return self.state

    def make_move(self, action: int):
        """ Make a move """
        self.state = self.state.make_move(action)
        return self.state

    @staticmethod
    def identities(state, actions_prob_dist: np.ndarray):
        """ Generate 8 symmetries """

        def make_identity(a, b):
            return State(a.ravel(), state.player), b.ravel()

        identities = []
        boards = state.boards.reshape((3,) + BOARD_SHAPE)
        apd = actions_prob_dist.reshape(BOARD_SHAPE)
        identities.append(make_identity(boards, apd))
        boards_m = np.flip(boards, 2)
        apd_m = np.fliplr(apd)
        identities.append(make_identity(boards_m, apd_m))
        for _ in range(3):
            boards = np.rot90(boards, 1, axes=(1, 2))
            apd = np.rot90(apd)
            identities.append(make_identity(boards, apd))
            boards_m = np.rot90(boards_m, 1, axes=(1, 2))
            apd_m = np.rot90(apd_m)
            identities.append(make_identity(boards_m, apd_m))
        return identities


class State:
    """ Contains board state, player who will make the next turn, game rules (allowed actions, end game test),
    as well as extra fields for MCTS and NN """

    def __init__(self, boards: np.ndarray, player: int):
        """
        :param boards: a stack of last 3 boards
        :param player: player to make turn
        """
        self.boards = boards
        self.board = boards[:BOARD_SIZE]  # current board
        self.player = player
        self.allowed_actions = self._get_allowed_actions()
        self.opponent_won = self._opponent_won()
        self.finished = self._is_finished()
        self.id = self._state_to_id()
        self.binary = self._get_binary()
        self.value = self._get_value()
        self.score = self._get_score()

    def __str__(self):
        s = ''
        board = self.board.reshape(BOARD_SHAPE)
        for row in board:
            s += ' '.join(PIECES[x] for x in row) + '\n'
        s += '-' * (BOARD_SIDE * 2 - 1)
        return s

    def log(self, logger):
        """ Print board to log """
        if logger.disabled:
            return
        board = self.board.reshape(BOARD_SHAPE)
        for row in board:
            logger.info(' '.join(PIECES[x] for x in row))
        logger.info('-' * (BOARD_SIDE * 2 - 1))

    def make_move(self, action):
        """ Make a turn """
        board = self.board.copy()
        board[action] = self.player
        boards = np.append(board, self.boards[:BOARD_SIZE * 2])
        state = State(boards, -self.player)
        return state

    def _get_allowed_actions(self):
        """ Get all actions that can be taken in current state """

        def is_valid_action(action):
            # Cell is not empty
            if self.board[action] != 0:
                return False
            # Border cell
            if action in BORDER_POSITIONS:
                return True
            # Cell has non-empty neighbor
            for x in NEIGHBORS[action]:
                if self.board[x]:
                    return True
            return False

        return list(filter(is_valid_action, INDEX_BOARD))

    def _opponent_won(self):
        """ Return True if opponent made a winning move """
        for w in WIN_POSITIONS:
            failed = False
            for i in w:
                if self.board[i] != -self.player:
                    failed = True
                    break
            if not failed:
                return True
        return False

    def _is_finished(self):
        """ Check end game conditions """
        return len(self.allowed_actions) == 0 or self.opponent_won

    def _state_to_id(self):
        """ Convert board state to id """
        return hash(self.board.tostring())

    def _get_binary(self):
        """ Create model input containing 3 last board states + board with all zeros/ones for current player """
        # Player 1 positions
        player1_positions = ZERO_BOARDS.copy()
        player1_positions[self.boards == 1] = 1
        # Player 2 positions
        player2_positions = ZERO_BOARDS.copy()
        player2_positions[self.boards == -1] = 1
        # Current player
        player = ZERO_BOARD if self.player == 1 else ONE_BOARD
        # Stack up player1 boards, player2 boards, and current player
        return np.concatenate([player1_positions, player2_positions, player])

    def _get_value(self):
        """ The value of the state for the current player, i.e. if the opponent played a winning move, you lose """
        if self.opponent_won:
            return -1
        return 0

    def _get_score(self):
        """ Score change for player and opponent """
        if self.opponent_won:
            return -1, 1
        return 0, 0
