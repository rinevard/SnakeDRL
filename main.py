import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from game.main_game import Game
from agent.base_agent import Agent
from agent.greedy_agent import GreedyAgent
from agent.dqn_agent import DQNAgent
from model.dqn_model import SnakeDQN
from common.helper import convert_screen_coord_to_grid
from common.game_elements import *


def play_with_agent(agent: Agent):
    game = Game(display_on=False)
    game_over_times = 0
    while True:
        if game_over_times % 1000 == 0:
            game.set_display_on()
        else:
            game.set_display_off()
        game_state = game.get_state()

        action = agent.get_action(game_state)
        game.step(action)

        new_game_state = game.get_state()

        if new_game_state.is_game_over():
            game_over_times += 1
            print(f"Times: {game_over_times}, Score: {new_game_state.get_score()}")
            game.reset()

def play_and_learn_with_dqn_agent(agent: DQNAgent, helper_agent: Agent=None, 
                                  display_rounds=100, helper_episodes=299):
    game = Game(display_on=False)
    game_over_times = 0
    episode_losses = []
    while True:
        if game_over_times % display_rounds == 0:
            game.set_display_on()
        else:
            game.set_display_off()

        # get (s, a, r, s', done) and update game
        game_state = game.get_state()



        # niko's test: use helper agent to benefit experience at first
        action = agent.get_action(game_state)
        if helper_agent and game_over_times <= helper_episodes:
            action = helper_agent.get_action(game_state)



        cur_score = game_state.get_score()
        game.step(action)
        next_state = game.get_state()
        next_score = next_state.get_score()
        
        head = game_state.get_snake_head()
        food = game_state.get_food()
        dis = abs(head[0] - food[0]) + abs(head[1] - food[1])
        next_head = next_state.get_snake_head()
        next_food = next_state.get_food()
        next_dis = abs(next_head[0] - next_food[0]) + abs(next_head[1] - next_food[1])
        reward = (next_score - cur_score) * 20 if (not next_state.is_game_over()) else -5
        reward += (dis - next_dis) if (dis - next_dis) > 0 else 0

        # learn
        agent.memorize(game_state, action, reward, next_state, next_state.is_game_over())
        loss = agent.learn()
        if loss:
            episode_losses.append(loss)

        if next_state.is_game_over():
            # print(f"State tensor: \n{next_state.get_state_tensor()[0]}")
            # print(f"Direction: {next_state.get_direction()}")
            # print(f"epsilon: {agent.epsilon}")
            game_over_times += 1
            print(f"Times: {game_over_times}, Score: {next_state.get_score()}")
            if episode_losses:
                print(f"Average loss for this episode: {sum(episode_losses) / len(episode_losses)}")
            episode_losses = []
            game.reset()


# greedy_agent
greedy_agent = GreedyAgent()

# dqn_agent
main_model = SnakeDQN(HEIGHT // BLOCK_SIZE, WIDTH // BLOCK_SIZE)
target_model = SnakeDQN(HEIGHT // BLOCK_SIZE, WIDTH // BLOCK_SIZE)
dqn_agent = DQNAgent(main_model, target_model)
play_and_learn_with_dqn_agent(dqn_agent)
