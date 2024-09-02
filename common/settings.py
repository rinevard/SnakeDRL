# Ensure that WIDTH and HEIGHT are multiples of BLOCKSIZE
GAME_WIDTH: int = 120
GAME_HEIGHT: int = 120
BLOCK_SIZE: int = 20

# for display settings
GAME_SPEED = 40

# for dqn_model
dqn_hidden_layer_size: int = 256

# for dqn_agent
learning_rate = 1.5e-4
gamma = 0.99
buffer_capacity = 10000
batch_size = 64
main_update_frequency = 4
target_update_frequency = 1000

# for playing
playing_display_rounds: int = 25
playing_total_rounds: int = 100

# for playing and learning
learning_total_episodes: int = 100000
learning_display_rounds: int = 300
epsilon_start: float = 0.8
epsilon_end: float = 0.025
epsilon_decay_steps = 10000