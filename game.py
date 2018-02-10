""" Game description classes can be replaced by any other game that complies to the same API """
import numpy as np

BOARD_SIZE = 7
ZERO_BOARD = np.zeros(BOARD_SIZE ** 2, dtype=np.int)
INDEX_BOARD = np.array([i for i in range(BOARD_SIZE ** 2)])
CELLS = {1: 'A', 0: '.', -1: 'B'}


def get_winners():
    """ Get all possible win combinations """

    def row_winners():
        """ All win combinations of the row """
        return [row[i:i + 4] for i in range(len(row) - 3)]

    def get_rows(board):
        """ All rows, columns and diagonals of the board """
        board = board.reshape(BOARD_SIZE, BOARD_SIZE)
        board_flip = np.fliplr(board)
        rows = [board[0]]
        cols = [board[:, 0]]
        diags1 = [board.diagonal(0)]
        diags2 = [board_flip.diagonal(0)]
        for i in range(1, BOARD_SIZE):
            rows.append(board[i])
            cols.append(board[:, i])
            if BOARD_SIZE - i >= 4:
                diags1.append(board.diagonal(i))
                diags1.append(board.diagonal(-i))
                diags2.append(board_flip.diagonal(i))
                diags2.append(board_flip.diagonal(-i))
        res = []
        res.extend(rows)
        res.extend(cols)
        res.extend(diags1)
        res.extend(diags2)
        return res

    winners = []
    for row in get_rows(INDEX_BOARD):
        winners.extend(row_winners())
    return winners


WINNERS = get_winners()


class Game:
    """ Four-in-a-row game. First player to get 4 marks in a straight line wins. Mark can be placed only on the
    border or in a cell that has a non-empty neighbor """

    def __init__(self):
        self.current_player = 1
        self.state = State(ZERO_BOARD.copy(), 1)
        self.name = 'four_in_a_row'
        self.grid_shape = (BOARD_SIZE, BOARD_SIZE)
        self.input_shape = (2, BOARD_SIZE, BOARD_SIZE)
        self.state_size = len(self.state.binary)
        self.action_size = len(ZERO_BOARD)

    def reset(self):
        """ Reset game """
        self.current_player = 1
        self.state = State(ZERO_BOARD.copy(), 1)
        return self.state

    def turn(self, action):
        """ Take turn"""
        next_state, value, finished = self.state.make_turn(action)
        self.state = next_state
        self.current_player = -self.current_player
        info = None
        return next_state, value, finished, info

    @staticmethod
    def identities(state, action_values):
        """ Original tuple (state, action_values) and its mirror images """
        # Original state
        identities = [(state, action_values)]
        # Horizontal mirror
        board = Game._mirror_horizontal(state.board)
        act_val = Game._mirror_horizontal(action_values)
        identities.append((State(board, state.player), act_val))
        # Vertical mirror
        board = Game._mirror_vertical(state.board)
        act_val = Game._mirror_vertical(action_values)
        identities.append((State(board, state.player), act_val))
        # Vertical and horizontal mirror
        board = np.flipud(state.board)
        act_val = np.flipud(action_values)
        identities.append((State(board, state.player), act_val))
        return identities

    @staticmethod
    def _mirror_horizontal(array):
        """ Horizontal mirror image of the array matrix """
        return np.ravel(np.fliplr(array.reshape(BOARD_SIZE, BOARD_SIZE)))

    @staticmethod
    def _mirror_vertical(array):
        """ Vertical mirror image of the array matrix """
        return np.ravel(np.flipud(array.reshape(BOARD_SIZE, BOARD_SIZE)))


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

    def make_turn(self, action):
        """ Make a turn """
        new_board = self.board.copy()
        new_board[action] = self.player
        new_state = State(new_board, -self.player)
        value = 0
        if new_state.finished:
            value = new_state.value[0]
        return new_state, value, new_state.finished

    def render(self, logger):
        """ Print board to the logger output """
        board = self.board.reshape(BOARD_SIZE, BOARD_SIZE)
        for row in board:
            logger.info(' '.join(CELLS[x] for x in row))
        logger.info('-' * (BOARD_SIZE * 2 - 1))

    def _get_allowed_actions(self):
        """ Get all actions that can be taken in current state """

        def is_valid_action(action):
            # Out of range
            if action < 0 or action >= len(self.board):
                return False
            # Cell is not empty
            if self.board[action] != 0:
                return False
            # Cell is next to the border
            if action < BOARD_SIZE \
                    or action >= len(self.board) - BOARD_SIZE \
                    or action % BOARD_SIZE == BOARD_SIZE - 1 \
                    or action % BOARD_SIZE == 0:
                return True
            # Cell has non-empty neighbor
            neighbors = [action - 1, action + 1, action - BOARD_SIZE, action + BOARD_SIZE, action - BOARD_SIZE - 1,
                         action - BOARD_SIZE + 1, action + BOARD_SIZE - 1, action + BOARD_SIZE + 1]
            return any(map(lambda x: self.board[x] != 0, neighbors))

        return list(filter(is_valid_action, INDEX_BOARD))

    def _did_opponent_win(self):
        """ Return True if opponent made a winning move """
        for w in WINNERS:
            if sum(self.board[w]) == 4 * -self.player:
                return True  # last turn resulted in a win combination
        return False

    def _is_finished(self):
        """ Check end game conditions """
        return len(self.allowed_actions) == 0 or self.opponent_won

    def _state_to_id(self):
        """ Convert board state to string id, which is a concatenation of two arrays of 0/1 corresponding to board
        positions occupied by each player """
        # Player 1 positions
        player1_positions = np.zeros(len(self.board), dtype=np.int)
        player1_positions[self.board == 1] = 1
        # Player 2 positions
        player2_positions = np.zeros(len(self.board), dtype=np.int)
        player2_positions[self.board == -1] = 1
        # Concatenate both arrays
        position = np.append(player1_positions, player2_positions)
        return ''.join(map(str, position))

    def _get_binary(self):
        """ Convert board state to a concatenation of two arrays of 0/1 corresponding to board positions occupied by
        current player and opponent """
        player_positions = np.zeros(len(self.board), dtype=np.int)
        player_positions[self.board == self.player] = 1
        opponent_positions = np.zeros(len(self.board), dtype=np.int)
        opponent_positions[self.board == -self.player] = 1
        return np.append(player_positions, opponent_positions)

    def _get_value(self):
        """ The value of the state for the current player, i.e. if the opponent played a winning move, you lose """
        if self.opponent_won:
            return -1, -1, 1
        return 0, 0, 0

    def _get_score(self):
        return self.value[1], self.value[2]
