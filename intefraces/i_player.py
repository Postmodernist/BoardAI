class IPlayer:
    """
    Interface for any type of player, either human or AI.
    """

    def get_name(self):
        """
        :return name: string
        """
        pass

    def get_action(self, state):
        """
        :param state: current game state
        :returns
            action: chosen action
            pi: action probability distribution
        """
        pass
