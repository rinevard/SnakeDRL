from common.game_elements import Action
from common.game_elements import State

class Agent():
    """
    An agent must define a getAction method.
    """
    def __init__(self) -> None:
        return
    
    def get_action(self, state: State=None) -> Action:
        return Action.GO_STRAIGHT