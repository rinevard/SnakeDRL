import torch
import torch.nn.functional as F
import random
from collections import deque

from common.settings import *
from common.utils import *
from agent.base_agent import LearningAgent
from model.dqn_model import *
from game.states import *


class ReplayBuffer():
    """
    Store experience whose type is: tuple[Tensor, int, float, Tensor, bool]
    """
    def __init__(self, capacity) -> None:
        self.buffer = deque(maxlen=capacity)

    def add(self, experience: tuple[torch.Tensor, int, float, torch.Tensor, bool]) -> None:
        """
        Add (s, a, r, s', done) into buffer.

        Note:
        There's no need to worry about overload.
        """
        # when the deque reaches its maxlen, 
        # it automatically removes the oldest item for new one.
        self.buffer.append(experience)
        return

    def sample(self, batch_size) -> list[tuple[torch.Tensor, int, float, torch.Tensor, bool]]:
        """
        Randomly sample and return a list with length 'batch_size' 
        of experiences (s, a, r, s', done).

        Note: 
        'batch_size' must be less or equal to the length of the buffer.
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class MLPAgent(LearningAgent):
    def __init__(self, device, 
                 learning_rate=learning_rate, gamma=gamma,
                 epsilon_start=epsilon_start, epsilon_end=epsilon_end, epsilon_decay_steps=epsilon_decay_steps, 
                 buffer_capacity=buffer_capacity, batch_size=batch_size, 
                 main_update_frequency=main_update_frequency, target_update_frequency=target_update_frequency, 
                 actions=[Action.TURN_RIGHT, Action.GO_STRAIGHT, Action.TURN_LEFT]):
        self.device = device
        print(f"Agent initialized on device: {self.device}")

        self.epsilon_decay_steps = (epsilon_start - epsilon_end) / epsilon_decay_steps
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.state_tensor_size = 15 # should match method '_get_state_tensor'
        self.main_model = SnakeMLP(self.state_tensor_size, len(actions)).to(self.device)
        self.target_model = SnakeMLP(self.state_tensor_size, len(actions)).to(self.device)
        self.optimizer = torch.optim.Adam(self.main_model.parameters(), lr=learning_rate)

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end

        self.current_steps = 0 # when self.current_steps % main_frequency == 0, update main; 
                               # when self.current_steps % target_frequency == 0, synchronize networks
        self.main_update_frequency = main_update_frequency
        self.target_update_frequency = target_update_frequency
        self.actions = actions
        super().__init__()

    def get_action(self, state: State = None) -> Action:
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            with torch.no_grad():
                # shape: (9, )
                state_tensor = self._get_state_tensor(state).to(device=self.device)
                # shape: (1, 3)
                q_values = self.main_model(state_tensor.unsqueeze(dim=0))
                return self.actions[torch.argmax(q_values).item()]
    
    def memorize(self, state: State, action: Action, reward: float, next_state: State, done: bool) -> None:
        """"
        Remember (s, a, r, s', done), which would be used for method 'learn'.
        """
        # save GPU memory
        s = self._get_state_tensor(state).cpu()
        a = convert_action_to_action_idx(action)
        r = reward
        s_new = self._get_state_tensor(next_state).cpu()
        exprience = (s, a, r, s_new, done)
        self.replay_buffer.add(exprience)
        return
    
    def learn(self) -> float:
        """
        Attempt to learn from collected experiences and return the loss. 
        If there are insufficient experiences, return None.
        """
        self.current_steps += 1
        if ((len(self.replay_buffer) < self.batch_size) or 
        (self.current_steps % self.main_update_frequency != 0)):
            return None
        
        # convert (s, a, r, s', done) into tensor type
        # tensor, int, float, tensor, bool
        expriences = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*expriences)
        states = torch.stack(states).to(device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device=self.device)
        next_states = torch.stack(next_states).to(device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device=self.device)

        # use 'gather' to get corresponding q-values
        predict_q_values: torch.Tensor = self.main_model(states).gather(dim=1, index=actions.unsqueeze(1))
        # use 'detach' to prevent gradient propagation 
        max_next_q_values: torch.Tensor = self.target_model(next_states).max(1).values.detach()
        target_q_values: torch.Tensor = rewards +  (1 - dones) * self.gamma * max_next_q_values
        
        loss = F.mse_loss(predict_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.main_model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        if self.current_steps >= self.target_update_frequency:
            self.current_steps = 0
            self._synchronize_networks()
        self._update_epsilon()
        return loss.item()
    
    def enter_train_mode(self, epsilon_start=epsilon_start, 
                        epsilon_end=epsilon_end, 
                        epsilon_decay_steps=epsilon_decay_steps):
        self.main_model.train()
        self.target_model.eval()
        self.epsilon = epsilon_start
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = (self.epsilon - self.epsilon_end) / self.epsilon_decay_steps
        return
    
    def enter_eval_mode(self) -> None:
        self.main_model.eval()
        self.target_model.eval()
        self.epsilon = 0
        self.epsilon_decay_steps = 0
        self.epsilon_end = 0
        return
    
    def save(self):
        self.main_model.save()

    def load(self) -> bool:
        return self.main_model.load(device=self.device) and self.target_model.load(device=self.device)
    
    def _synchronize_networks(self) -> None:
        self.target_model.load_state_dict(self.main_model.state_dict())
        self.target_model.eval()
        return
    
    def _update_epsilon(self) -> None:
        self.epsilon -= self.epsilon_decay_steps
        if self.epsilon <= self.epsilon_end:
            self.epsilon = self.epsilon_end
        return

    def _get_state_tensor(self, state: State) -> torch.Tensor:
        """
        Return a tensor representation of the current game state

        Returns:
        A tensor with shape (15,)

        idx:
        direction_x: 0
        direction_y: 1
        food_relative_pos: 2, 3
        snake_len: 4
        steps_of_do_nothing: 5

        right_danger_dis_after_turn_right: 6
        straight_danger_dis_after_turn_right: 7
        left_danger_dis_after_turn_right: 8
        
        right_danger_dis_after_go_straight: 9
        straight_danger_dis_after_go_straight: 10
        left_danger_dis_after_go_straight: 11

        right_danger_dis_after_turn_left: 12
        straight_danger_dis_after_turn_left: 13
        left_danger_dis_after_turn_left: 14
        """
        head: tuple[int, int] = state.get_snake_head()
        body: list[tuple[int, int]] = state.get_snake_body()
        food: tuple[int, int] = state.get_food()

        direction: Direction = state.get_direction()
        # direction_x, direction_y
        direction_tuple: tuple[int, int] = convert_direction_to_tuple(direction)
        # food_relative_pos
        food_relative_pos: tuple[int, int] = convert_global_pos_to_relative_pos(head, 
                                                               direction, 
                                                               food)

        # snake_len
        snake_length: int = len(state.get_snake())
        # steps_of_do_nothing
        steps_of_do_nothing: int = state.steps_of_do_nothing

        # danger_distances
        straight_direction: Direction = state.direction
        left_direction: Direction = change_direction(straight_direction, Action.TURN_LEFT)
        right_direction: Direction = change_direction(straight_direction, Action.TURN_RIGHT)

        right_direction_tuple: tuple[int, int] = convert_direction_to_tuple(right_direction)
        straight_direction_tuple: tuple[int, int] = convert_direction_to_tuple(straight_direction)
        left_direction_tuple: tuple[int, int] = convert_direction_to_tuple(left_direction)
        
        right_danger_dis, straight_danger_dis, left_danger_dis = state.get_closest_danger_distances(head, 
                                                              straight_direction)

        head_after_turn_right: tuple[int, int] = (head[0] + right_direction_tuple[0], 
                                                 head[1] + right_direction_tuple[1])
        head_after_go_straight: tuple[int, int] = (head[0] + straight_direction_tuple[0], 
                                                   head[1] + straight_direction_tuple[1])
        head_after_turn_left: tuple[int, int] = (head[0] + left_direction_tuple[0], 
                                                 head[1] + left_direction_tuple[1])
        
        danger_distances_after_turn_right = state.get_closest_danger_distances(head_after_turn_right, 
                                                                              right_direction)
        danger_distances_after_go_straight = state.get_closest_danger_distances(head_after_go_straight, 
                                                                               straight_direction)
        danger_distances_after_turn_left = state.get_closest_danger_distances(head_after_turn_left, 
                                                                             left_direction)

        # shape: (15,)
        # dtype: torch.float32
        return torch.tensor([
            direction_tuple[0], 
            direction_tuple[1], 
            food_relative_pos[0], 
            food_relative_pos[1], 
            snake_length, 
            steps_of_do_nothing, 
            danger_distances_after_turn_right[0], 
            danger_distances_after_turn_right[1], 
            danger_distances_after_turn_right[2],
            danger_distances_after_go_straight[0], 
            danger_distances_after_go_straight[1], 
            danger_distances_after_go_straight[2],
            danger_distances_after_turn_left[0], 
            danger_distances_after_turn_left[1], 
            danger_distances_after_turn_left[2], 
        ]).to(dtype=torch.float32)
