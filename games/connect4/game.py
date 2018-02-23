import numpy as np

# Board
BOARD_SHAPE = (6, 7)
BOARD_SIZE = BOARD_SHAPE[0] * BOARD_SHAPE[1]
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
        for i in range(1, BOARD_SHAPE[1]):
            if i < BOARD_SHAPE[0]:
                rows.append(board[i])
            cols.append(board[:, i])
            if len(board.diagonal(i)) >= 4:
                diagonals1.append(board.diagonal(i))
                diagonals2.append(board_flip.diagonal(i))
            if len(board.diagonal(-i)) >= 4:
                diagonals1.append(board.diagonal(-i))
                diagonals2.append(board_flip.diagonal(-i))
        return rows + cols + diagonals1 + diagonals2

    win_positions = []
    for row in get_rows(INDEX_BOARD):
        win_positions.extend([row[j:j + 4] for j in range(len(row) - 3)])
    return win_positions


WIN_POSITIONS = get_win_positions()


class Game:
    """ Connect 4 game. First player to get 4 marks in a straight line wins. Mark can be placed only on the bottom edge
    or on top of the other marks """

    name = 'connect4'
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
    def identities(state, actions_prob_dist):
        """ Generate 2 symmetries """

        def make_identity(x, y):
            return State(x.ravel(), state.player), y.ravel()

        identities = []
        board = state.board.reshape(BOARD_SHAPE)
        apd = actions_prob_dist.reshape(BOARD_SHAPE)
        identities.append(make_identity(board, apd))
        board_m = np.fliplr(board)
        apd_m = np.fliplr(apd)
        identities.append(make_identity(board_m, apd_m))
        return identities


class State:
    """ Contains board state, player who will make the next turn, game rules (allowed actions, end game test),
    as well as extra fields for MCTS and NN """

    def __init__(self, board, player):
        self.board = board
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
        s += '-' * (BOARD_SHAPE[1] * 2 - 1)
        return s

    def log(self, logger):
        """ Print board to log """
        board = self.board.reshape(BOARD_SHAPE)
        for row in board:
            logger.info(' '.join(PIECES[x] for x in row))
        logger.info('-' * (BOARD_SHAPE[1] * 2 - 1))

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
            # Cell is on the bottom edge
            if action >= BOARD_SIZE - BOARD_SHAPE[1]:
                return True
            # Cell is on top of occupied cell
            return self.board[action + BOARD_SHAPE[1]] != 0

        return list(filter(is_valid_action, INDEX_BOARD))

    def _opponent_won(self):
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
