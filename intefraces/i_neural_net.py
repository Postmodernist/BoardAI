class INeuralNet:
    """
    This class specifies the base NeuralNet class. To define your own neural
    network, subclass this class and implement the functions below. The neural
    network does not consider the current player, and instead only deals with
    the canonical form of the board.
    """

    @staticmethod
    def create():
        """
        :return a new INeuralNet object
        """
        pass

    def train(self, examples):
        """
        This function trains the neural network with examples obtained from
        self-play.
        :param examples: a list of training examples, where each example is of
            form (board, pi, v). pi is the MCTS informed policy vector for the
            given board, and v is its value. The board is in canonical form
        """
        pass

    def predict(self, canonical_board, valid_actions):
        """
        :param canonical_board: current board in its canonical form
        :param valid_actions: a list of valid actions
        :returns
            pi: a policy vector for the current board, a numpy array of length
                game.ACTION_SIZE
            v: a float in [-1,1] that gives the value of the current board
        """
        pass

    def save(self, folder, file_name):
        """
        Saves the current neural network (with its parameters) in folder/file_name
        """
        pass

    def load(self, folder, file_name):
        """
        Loads parameters of the neural network from folder/file_name
        """
        pass
