from common.settings import *
from common.utils import *
from game.states import *
from agent.base_agent import Agent

class GreedyAgent(Agent):
    def __init__(self) -> None:
        super().__init__()

    def get_action(self, state: State=None) -> Action:
        if not state:
            return 0
        
        food = state.get_food()
        direction = state.get_direction()
        head = state.get_snake_head()

        relative_vec = convert_global_pos_to_relative_pos(head, direction, food)
        # Move horizontally until the x-coordinate aligns with the food, then move vertically

        # Note that x-axis is 90 degrees **counterclockwise** from the y-axis 
        if relative_vec[0] > 0:
            return Action.TURN_LEFT
        elif relative_vec[0] < 0:
            return Action.TURN_RIGHT
        elif relative_vec[1] < 0 :
            return Action.TURN_RIGHT
        else:
            return Action.GO_STRAIGHT