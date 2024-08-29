from common.game_elements import Action
from common.helper import global_pos_to_relative_pos
"""
state:
1. 豆子相对自己方向的位置。这样一来，就没有上下左右之分了，只有“面朝的方向”。
2. 选择一个action是否会撞墙
3. 自己的身体相对自己方向的位置
"""

from .base_agent import Agent



class QlearningAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.q_value_dict = {}

    def get_action(self, state=None) -> Action:
        return super().get_action(state)
    
    def train(origin_state, action, reward, new_state):
        return
    
