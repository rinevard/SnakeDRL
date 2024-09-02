# Ensure that WIDTH and HEIGHT are multiples of BLOCKSIZE
WIDTH: int = 120
HEIGHT: int = 120
BLOCK_SIZE: int = 20

# for display settings
SPEED = 40

# for dqn_model
hidden_layer_size: int = 256

# for dqn_agent
learning_rate = 1.5e-4
gamma = 0.99
buffer_capacity = 10000
batch_size = 64
main_update_frequency = 4
target_update_frequency = 1000

# for playing
display_rounds_when_playing: int = 25
total_rounds_when_playing: int = 100

# for playing and learning
total_episodes_when_learning: int = 100000
display_rounds_when_learning: int = 300
epsilon_start: float = 0.8
epsilon_end: float = 0.025
time_for_epsilon_to_delay_to_end = 10000