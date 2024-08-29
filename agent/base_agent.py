from common.game_elements import Action

class Agent():
    """
    An agent must define a getAction method.
    """
    def __init__(self) -> None:
        return
    
    def get_action(self, state=None) -> Action:
        """
        state: A dictionary containing the following keys:
            - 'snake' (list of tuple): List of coordinates representing the snake's body.
            Each element is a tuple (x: int, y: int). snake[0] is the position of the snake's head.
            - 'direction' (tuple): x = 1 if right else -1; y = 1 if down else -1.
            - 'food' (tuple): Coordinates of the food as a tuple (x: int, y: int).
            - 'score' (int)
            - 'is_game_over' (bool)
        """
        # TODO: change 'state' and 'action' to classes
        return Action.GO_STRAIGHT