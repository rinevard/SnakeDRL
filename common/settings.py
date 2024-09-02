# Ensure that WIDTH and HEIGHT are multiples of BLOCKSIZE
GAME_WIDTH: int = 640
GAME_HEIGHT: int = 480
BLOCK_SIZE: int = 20

# for display settings
GAME_SPEED = 30

# for dqn_model
dqn_hidden_layer_size: int = 256

# for dqn_agent
learning_rate = 1.5e-4
gamma = 0.99
buffer_capacity = 2000
batch_size = 64
main_update_frequency = 4
target_update_frequency = 1000

# for playing
playing_total_rounds: int = 50
playing_rounds_per_display: int = 25

# for playing and learning
learning_total_episodes: int = 3000
learning_episodes_per_display: int = 300
learning_episodes_per_save: int = 50
epsilon_start: float = 0.8
epsilon_end: float = 0
epsilon_decay_steps = 4000

# for game
addtional_score_when_winning: int = 100