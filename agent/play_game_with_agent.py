import matplotlib.pyplot as plt
from collections import deque

from agent.base_agent import Agent
from agent.dqn_agent import DQNAgent
from game.main_game import Game
from common.constants_and_enums import *
from game.states import *

def reward_func(state: State, action: Action, next_state: State):
    if (next_state.is_game_over()):
        return -300 / state.get_snake_length()
    
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
        reward = (next_score - cur_score) * 20 + (0.5 * state.get_snake_length())
    return reward

def play_with_agent(agent: DQNAgent, 
                    display_rounds: int=25, total_rounds: int=100):
    print(f"Play for {total_rounds} rounds and coumpute the average score...")
    agent.enter_eval_mode()
    game = Game(display_on=False)
    game_over_times = 0
    scores = []
    while game_over_times < total_rounds:
        if game_over_times % display_rounds == 0:
            game.set_display_on()
        else:
            game.set_display_off()
        game_state = game.get_state()

        action = agent.get_action(game_state)
        game.step(action)

        new_game_state = game.get_state()

        if new_game_state.is_game_over():
            game_over_times += 1
            print(f"Times: {game_over_times}, Score: {new_game_state.get_score()}\n")
            scores.append(new_game_state.get_score())
            game.reset()
    
    print(f"\nHighest score in {total_rounds} rounds: {max(scores)}")
    print(f"\nAverage score in {total_rounds} rounds: {sum(scores) / total_rounds}\n")
    return

def play_and_learn_with_dqn_agent(agent: DQNAgent, total_episodes: int=99999, 
                                  update_plot_callback=None, 
                                  helper_agent: Agent=None, 
                                  display_rounds=300, helper_episodes=59, 
                                  epsilon_start=0.8, epsilon_end=0.025,
                                  epsilon_decay_steps=10000):
    """
    Parameters:
        update_plot_callback: a callback function updating the plot, (float, int) -> None
    """
    agent.enter_train_mode(epsilon_start=epsilon_start, 
                           epsilon_end=epsilon_end,
                           epsilon_decay_steps=epsilon_decay_steps)
    game = Game(display_on=False)
    game_over_times = 0
    episode_losses = []
    avg_loss = None
    avg_score = None
    scores_recent_hundred_round = deque(maxlen=100)
    agent.main_model.train()
    agent.target_model.eval()
    while game_over_times < total_episodes:
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
            print(f"Feel free to press Ctrl+C to terminate the program or close the window")

            # compute average loss and average score 
            scores_recent_hundred_round.append(next_state.get_score())


            print(f"Play times: {game_over_times}, Score: {next_state.get_score()}")
            print(f"Epsilon: {agent.epsilon}")
            if episode_losses:
                avg_loss = sum(episode_losses) / len(episode_losses)
                print(f"Average loss: {avg_loss}")
            if scores_recent_hundred_round:
                avg_score = sum(scores_recent_hundred_round) / len(scores_recent_hundred_round)
                print(f"Average score current 100 rounds: {avg_score}")
            print('\n')

            # update the plot
            if update_plot_callback and avg_loss and avg_score:
                update_plot_callback(avg_loss, avg_score)

            # reset
            episode_losses = []
            game.reset()
            

def create_training_plotter():
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

    # add hspace to subplots to avoid overlap
    plt.subplots_adjust(hspace=0.4)

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