import numpy as np

# Board
BOARD_SIDE = 7
BOARD_SIZE = BOARD_SIDE ** 2
BOARD_SHAPE = (BOARD_SIDE, BOARD_SIDE)
ZERO_BOARD = np.zeros(BOARD_SIZE, dtype=np.int)
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
        win_positions.extend([row[i:i + 4] for i in range(len(row) - 3)])
    return win_positions


WIN_POSITIONS = get_win_positions()


class Game:
    """ Four-in-a-row game. First player to get 4 marks in a straight line wins. Mark can be placed only on the
    border or in a cell that has a non-empty neighbor. Game class can be replaced by any other game that complies
    to the same API """

    name = 'four_in_a_row'
    board_size = BOARD_SIZE
    board_shape = BOARD_SHAPE
    input_shape = (2,) + BOARD_SHAPE
    pieces = PIECES

    def __init__(self):
        self.state = State(ZERO_BOARD.copy(), 1)

    def reset(self):
        """ Reset game """
        self.state = State(ZERO_BOARD.copy(), 1)
        return self.state

    def make_move(self, action):
        """ Make a move """
        self.state = self.state.make_move(action)
        return self.state

    @staticmethod
    def identities(state, actions_values):
        """ Generate 8 symmetries of state and actions_values arrays """

        def make_identity(x, y):
            return State(x.ravel(), state.player), y.ravel()

        identities = [(state, actions_values)]
        board = state.board.reshape(BOARD_SHAPE)
        av = actions_values.reshape(BOARD_SHAPE)
        board_m = np.fliplr(board)
        av_m = np.fliplr(av)
        identities.append(make_identity(board_m, av_m))
        for _ in range(3):
            board = np.rot90(board)
            av = np.rot90(av)
            identities.append(make_identity(board, av))
            board_m = np.rot90(board_m)
            av_m = np.rot90(av_m)
            identities.append(make_identity(board_m, av_m))
        return identities


class State:
    """ Contains board state, player who will make the next turn, game rules (allowed actions, end game test),
    as well as extra fields for MCTS and NN """

    def __init__(self, board, player):
        self.board = board
        self.player = player
        self.allowed_actions = self._get_allowed_actions()
        self.opponent_won = self._did_opponent_win()
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
        board = self.board.reshape(BOARD_SHAPE)
        for row in board:
            logger.info(' '.join(PIECES[x] for x in row))
        logger.info('-' * (BOARD_SIDE * 2 - 1))

    def make_move(self, action):
        """ Make a turn """
        board = self.board.copy()
        board[action] = self.player
        state = State(board, -self.player)
        return state

    def _get_allowed_actions(self):
        """ Get all actions that can be taken in current state """

        def is_valid_action(action):
            # Out of range
            if action < 0 or action >= BOARD_SIZE:
                return False
            # Cell is not empty
            if self.board[action] != 0:
                return False
            # Cell is next to the border
            if action < BOARD_SIDE \
                    or action >= len(self.board) - BOARD_SIDE \
                    or action % BOARD_SIDE == BOARD_SIDE - 1 \
                    or action % BOARD_SIDE == 0:
                return True
            # Cell has non-empty neighbor
            neighbors = [action - 1, action + 1, action - BOARD_SIDE, action + BOARD_SIDE, action - BOARD_SIDE - 1,
                         action - BOARD_SIDE + 1, action + BOARD_SIDE - 1, action + BOARD_SIDE + 1]
            return any(map(lambda x: self.board[x] != 0, neighbors))

        return list(filter(is_valid_action, INDEX_BOARD))

    def _did_opponent_win(self):
        """ Return True if opponent made a winning move """
        for w in WIN_POSITIONS:
            if sum(self.board[w]) == 4 * -self.player:
                return True  # last turn resulted in a win combination
        return False

    def _is_finished(self):
        """ Check end game conditions """
        return len(self.allowed_actions) == 0 or self.opponent_won

    def _state_to_id(self):
        """ Convert board state to string id, which is a concatenation of two arrays of 0/1 corresponding to board
        positions occupied by player A and player B """
        # Player 1 positions
        player1_positions = ZERO_BOARD.copy()
        player1_positions[self.board == 1] = 1
        # Player 2 positions
        player2_positions = ZERO_BOARD.copy()
        player2_positions[self.board == -1] = 1
        # Concatenate both arrays
        positions = np.append(player1_positions, player2_positions)
        return ''.join(map(str, positions))

    def _get_binary(self):
        """ Convert board state to a concatenation of two arrays of 0/1 corresponding to board positions occupied by
        current player and opponent """
        player_positions = ZERO_BOARD.copy()
        player_positions[self.board == self.player] = 1
        opponent_positions = ZERO_BOARD.copy()
        opponent_positions[self.board == -self.player] = 1
        return np.append(player_positions, opponent_positions)

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
