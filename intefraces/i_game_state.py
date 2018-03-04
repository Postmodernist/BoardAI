class IGameState:
    """
    Represents a state of the game. Must be implemented along with IGame interface.
    """

    def get_board(self):
        """
        :return: current board
        """
        pass

    def get_canonical_board(self):
        """
        :return canonical_board: returns canonical form of board. The canonical form should be
            independent of player. For e.g. in chess, the canonical form can be chosen to be from
            the pov of white. When the player is white, we can return board as is. When the player
            is black, we can invert the colors and return the board
        """
        pass

    def get_player(self):
        """
        :return current player
        """
        pass

    def get_turn(self):
        """
        :return current turn
        """
        pass

    def get_valid_actions(self):
        """
        :return valid_actions: a binary vector of length ACTION_SIZE, 1 for moves that are
            valid from the current board and player, 0 for invalid moves
        """
        pass

    def get_next_state(self, action):
        """
        :param action: action taken by current player
        :return next_state: new game state after applying action
        """
        pass

    def is_game_finished(self):
        """
        :return is_finished: True if game is over, False otherwise
        """
        pass

    def get_value(self):
        """
        :return value: -1 if player lost, 0 if game draw
        """
        pass

    def get_hashable(self):
        """
        :return a hashable representation of current board. Required by MCTS
        """
        pass

    def log(self, logger):
        """
        Write current board state to log
        :param logger: logger to use
        """
        pass
