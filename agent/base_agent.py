from common.settings import *
from game.states import *

class Agent():
    """
    An agent must define a 'get_action' method.
    """
    def __init__(self) -> None:
        return
    
    def get_action(self, state: State=None) -> Action:
        return Action.GO_STRAIGHT
    
class LearningAgent(Agent):
    """
    An agent must define a 'get_action' method, a 'memorize' method, an 'learn' method.
    """
    def __init__(self) -> None:
        return

    def get_action(self, state: State = None) -> Action:
        return Action.GO_STRAIGHT
    
    def memorize(self, state: State, action: Action, reward: float, next_state: State):
        return

    def learn(self):
        return