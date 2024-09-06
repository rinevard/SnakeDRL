# Ensure that WIDTH and HEIGHT are multiples of BLOCKSIZE
GAME_WIDTH: int = 300
GAME_HEIGHT: int = 300
BLOCK_SIZE: int = 30

# for display settings
GAME_SPEED = 20

# for dqn_model
# To change model weights, place your .pth file in the '/model/weights' directory 
# and rename it to either 'cnn_model.pth' or 'mlp_model.pth'
mlp_hidden_layer_size: int = 256

cnn_conv_mid_channels: int = 256
cnn_out_channels: int = 128
cnn_fc_hidden_layer_size: int = 256

# for dqn_agent
learning_rate: float = 1e-4
gamma: float = 0.99
buffer_capacity: int = 1000
batch_size: int = 64
main_update_frequency: int = 4
target_update_frequency: int = 1000

# for playing
playing_total_rounds: int = 100
playing_rounds_per_display: int = 25

# for playing and learning
learning_total_episodes: int = 3000
learning_episodes_per_display: int = 300
learning_episodes_per_save: int = 50
epsilon_start: float = 0.8  
epsilon_end: float = 0
epsilon_decay_steps: int = 4000

# for game
addtional_score_when_winning: int = 100