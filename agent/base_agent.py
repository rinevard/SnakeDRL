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
    An agent must define a 'get_action' method, 
    a 'memorize' method, an 'learn' method, 
    an 'enter_train_mode', an 'enter_eval_mode' method.
    and 'save' method,  'load' method, 
    """
    def __init__(self) -> None:
        return

    def get_action(self, state: State = None) -> Action:
        return Action.GO_STRAIGHT
    
    def memorize(self, state: State, action: Action, reward: float, next_state: State):
        return

    def learn(self) -> float:
        """
        Attempt to learn from collected experiences and return the loss. 
        If there are insufficient experiences, return None.
        """
        return
    
    def enter_train_mode(self):
        return
    
    def enter_eval_mode(self):
        return
    
    def save(self):
        return
    
    def load(self):
        return