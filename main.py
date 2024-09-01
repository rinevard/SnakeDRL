import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

from game.main_game import Game
from agent.base_agent import Agent
from agent.greedy_agent import GreedyAgent
from agent.dqn_agent import DQNAgent
from model.dqn_model import SnakeLinerDQN
from common.game_elements import *
from common.states import *
from common.utils import *

def reward_func(state: State, action: Action, next_state: State):
    if (next_state.is_game_over()):
        return -10
    
    reward = 0

    # encourage get closer to food
    head = state.get_snake_head()
    food = state.get_food()
    dis = abs(head[0] - food[0]) + abs(head[1] - food[1])
    next_head = next_state.get_snake_head()
    next_food = next_state.get_food()
    next_dis = abs(next_head[0] - next_food[0]) + abs(next_head[1] - next_food[1])
    reward += (dis - next_dis)

    # score reward
    cur_score = state.get_score()
    next_score = next_state.get_score()
    if (next_score != cur_score):
        reward = (next_score - cur_score) * 20
    return reward




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

def play_and_learn_with_dqn_agent(agent: DQNAgent, update_plot_callback=None, 
                                  helper_agent: Agent=None, 
                                  display_rounds=100, helper_episodes=59):
    """
    Parameters:
        update_plot_callback: a callback function updating the plot, (float, int) -> None
    """
    game = Game(display_on=False)
    game_over_times = 0
    episode_losses = []
    scores_recent_hundred_round = deque(maxlen=100)
    agent.main_model.train()
    agent.target_model.eval()
    while True:
        if game_over_times % display_rounds == 0:
            game.set_display_on()
        else:
            game.set_display_off()

        # get (s, a, r, s', done) and update game
        game_state = game.get_state()

        # use helper agent to benefit experience at first
        # does it work? i dont know =)
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
            game_over_times += 1

            # save weights
            agent.main_model.save()

            # compute average loss and average score 
            scores_recent_hundred_round.append(next_state.get_score())
            avg_loss = sum(episode_losses) / len(episode_losses)
            avg_score = sum(scores_recent_hundred_round) / len(scores_recent_hundred_round)

            print(f"Play times: {game_over_times}, Score: {next_state.get_score()}")
            if episode_losses:
                print(f"Average loss: {avg_loss}")
            print(f"Epsilon: {agent.epsilon}")
            print(f"Average score current 100 rounds: {avg_score}")
            print('\n')

            # update the plot
            if update_plot_callback:
                update_plot_callback(avg_loss, avg_score)

            # reset
            episode_losses = []
            game.reset()
            

def create_plotter():
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))
    loss_line, = ax1.plot([], [], 'r-')
    score_line, = ax2.plot([], [], 'b-')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Average score in 100 rounds')
    ax2.set_title('Game Score')

    all_losses = []
    all_scores = []

    def update_plot(loss, score):
        all_losses.append(loss)
        all_scores.append(score)
        loss_line.set_data(range(len(all_losses)), all_losses)
        score_line.set_data(range(len(all_scores)), all_scores)
        ax1.relim()
        ax1.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()

    return update_plot

if __name__ == "__main__":
    # create a callback function to update the figure dynamically
    update_plot = create_plotter()

    # greedy_agent
    greedy_agent = GreedyAgent()

    # dqn_agent
    main_model = SnakeLinerDQN(9, 3)
    target_model = SnakeLinerDQN(9, 3)
    dqn_agent = DQNAgent(main_model, target_model)

    while True:
        user_input = input("Start training from scratch? (y/n): ").lower()
        if user_input == 'y':
            print("Starting training from scratch...")
            break
        elif user_input == 'n':
            print("Attempting to load previous model...")
            if dqn_agent.main_model.load() and dqn_agent.target_model.load():
                dqn_agent.epsilon = 0.1
                print("Previous model loaded successfully.")
            else:
                print("Failed to load previous model. Starting from scratch...")
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

    play_and_learn_with_dqn_agent(dqn_agent, update_plot)
