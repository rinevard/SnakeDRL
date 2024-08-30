import torch

from common.game_elements import Action
from common.helper import state_dic_to_state_tensor
from model.dqn_model import SnakeDQN
from .base_agent import Agent

class DQNlearningAgent(Agent):
    def __init__(self, grid_width, grid_height) -> None:
        super().__init__()
        self.dqn = SnakeDQN(grid_width, grid_height)

    def get_action(self, state=None) -> Action:
        actions = [Action.TURN_RIGHT, Action.GO_STRAIGHT, Action.TURN_LEFT]

        if not state:
            return actions[1]
        state_tensor = state_dic_to_state_tensor(state)
        q_values = self.dqn(state_tensor)

        max_q_idx = torch.argmax(q_values).item()
        return actions[max_q_idx]

    def train(origin_state, action, reward, new_state):
        return
