import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from game.main_game import Game
from agent.base_agent import Agent
from agent.greedy_agent import GreedyAgent
from agent.dqn_agent import DQNAgent
from model.dqn_model import *
from common.helper import convert_screen_coord_to_grid
from common.helper import *


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

        game.step(action)
        next_state = game.get_state()
        reward = reward_func(game_state, action, next_state)

        # learn
        agent.memorize(game_state, action, reward, next_state, next_state.is_game_over())
        loss = agent.learn()
        if loss:
            episode_losses.append(loss)

        if next_state.is_game_over():
            # print(f"State tensor: \n{next_state.get_state_tensor()}")
            # print(f"Direction: {next_state.get_direction()}")
            # print(f"epsilon: {agent.epsilon}")
            game_over_times += 1
            print(f"Times: {game_over_times}, Score: {next_state.get_score()}")
            if episode_losses:
                print(f"Average loss for this episode: {sum(episode_losses) / len(episode_losses)}")
            episode_losses = []
            game.reset()























from game.main_game import TestGame
def play_and_learn_with_dqn_agent_in_test_game(agent: DQNAgent, helper_agent: Agent=None, 
                                  display_rounds=200, helper_episodes=299):
    game = TestGame(display_on=False)
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

        game.step(action)
        next_state = game.get_state()
        reward = reward_func(game_state, action, next_state)

        # if reward != 0:
        #     print(game_state.get_state_tensor())
        #     print(action)
        #     print(reward)
        #     print(next_state.get_state_tensor())

        # learn
        agent.memorize(game_state, action, reward, next_state, next_state.is_game_over())
        loss = agent.learn()
        if loss:
            episode_losses.append(loss)

        if next_state.is_game_over():
            game_over_times += 1
            print(f"Times: {game_over_times}, Score: {next_state.get_score()}")
            if episode_losses:
                print(f"Average loss for this episode: {sum(episode_losses) / len(episode_losses)}")
            # print(f"State tensor: \n{next_state.get_state_tensor()[0]}")
            # print(f"Direction: {next_state.get_direction()}")
            print(f"epsilon: {agent.epsilon}")

            episode_losses = []
            game.reset()


def reward_func(state: State, action: Action, next_state: State):
    if (next_state.is_game_over()):
        return -4
    
    reward = 0

    # encourage get closer to food
    head = state.get_snake_head()
    food = state.get_food()
    dis = abs(head[0] - food[0]) + abs(head[1] - food[1])
    next_head = next_state.get_snake_head()
    next_food = next_state.get_food()
    next_dis = abs(next_head[0] - next_food[0]) + abs(next_head[1] - next_food[1])
    reward += ((dis - next_dis) / (HEIGHT // BLOCK_SIZE))

    # score reward
    cur_score = state.get_score()
    next_score = next_state.get_score()
    if (next_score != cur_score):
        reward = (next_score - cur_score) * 15
    return reward




# greedy_agent
greedy_agent = GreedyAgent()

# dqn_agent
main_model = SnakeLinerDQN(9, 3)
target_model = SnakeLinerDQN(9, 3)
dqn_agent = DQNAgent(main_model, target_model)
play_and_learn_with_dqn_agent(dqn_agent)