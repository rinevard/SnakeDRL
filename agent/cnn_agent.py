import torch
import torch.nn.functional as F
import random

from common.settings import *
from common.utils import *
from agent.base_agent import LearningAgent
from agent.mlp_agent import ReplayBuffer
from model.dqn_model import *
from game.states import *

class CNNAgent(LearningAgent):
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
        self.conv_input_channels = 4 # should match method '_get_state_tensor'
        self.main_model = SnakeCNN(self.conv_input_channels, len(actions)).to(self.device)
        self.target_model = SnakeCNN(self.conv_input_channels, len(actions)).to(self.device)
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
                # shape: (4, grid_height + 2, grid_width + 2)
                # note that 'grid_height' comes first
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
        A tensor with shape (4, grid_height + 2, grid_width + 2), note that 'grid_height' comes first

        First channel is dangerous areas, including edges and snake body.

        Second channel is food.
        
        Third channel is snake head and current direction.
        
        Forth channel is snake itself.
        """
        head: tuple[int, int] = state.get_snake_head()
        body: list[tuple[int, int]] = state.get_snake_body()
        food: tuple[int, int] = state.get_food()

        direction: Direction = state.get_direction()
        # direction_x, direction_y
        direction_tuple: tuple[int, int] = convert_direction_to_tuple(direction)

        grid_height = state.get_grid_height()
        grid_width = state.get_grid_width()
        head: tuple[int, int] = state.get_snake_head()
        body: list[tuple[int, int]] = state.get_snake_body()
        food: tuple[int, int] = state.get_food()

        direction: Direction = state.get_direction()

        grid_height = state.get_grid_height()
        grid_width = state.get_grid_width()

        # Initialize tensor with zeros
        # x in [0, grid_width + 2]; y in [0, grid_height + 2]
        state_tensor = torch.zeros((4, grid_height + 3, grid_width + 3), dtype=torch.float32)

        # First channel: dangerous areas (edges and snake body)
        state_tensor[0, 0, :] = 1  # Top edge
        state_tensor[0, -1, :] = 1  # Bottom edge
        state_tensor[0, :, 0] = 1  # Left edge
        state_tensor[0, :, -1] = 1  # Right edge
        for x, y in body:
            state_tensor[0, y + 1, x + 1] = 1  # Snake body

        # Second channel: food
        food_x, food_y = food
        state_tensor[1, food_y + 1, food_x + 1] = 1

        # Third channel: snake head and direction
        head_x, head_y = head
        state_tensor[2, head_y + 1, head_x + 1] = 1
        dir_x, dir_y = direction_tuple
        state_tensor[2, head_y + 1 + dir_y, head_x + 1 + dir_x] = 0.5
    
        # Fourth channel: snake itself
        for x, y in body:
            state_tensor[3, y + 1, x + 1] = 0.5
        state_tensor[3, head_y + 1, head_x + 1] = 1

        # Return:
        # shape: (4, grid_height + 2, grid_width + 2)
        # dtype: torch.float32

        # Note:
        # If you with to change the shape of returned tensor, 
        # please also change 'self.conv_input_channels'
        return state_tensor.to(dtype=torch.float32)