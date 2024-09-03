import torch

from common.settings import *
from agent.play_game_with_agent import *
from agent.dqn_agent import DQNAgent
from model.dqn_model import SnakeLinerDQN

def main():
    """
    Start
    │
    └─── Choose mode
        │
        ├─── Play
        │   │
        │   └─── Load previous model
        │       │
        │       ├─── Success
        │       │   └─── Play with loaded model
        │       │
        │       └─── Fail
        │           │
        │           └─── Continue without loaded model?
        │               │
        │               ├─── Yes
        │               │   └─── Play with untrained model
        │               │
        │               └─── No
        │                   └─── Return to mode selection
        │
        └─── Learn and Play
            │
            └─── Start training from scratch?
                │
                ├─── Yes
                │   └─── Start new training
                │
                └─── No
                    │
                    └─── Load previous model
                        │
                        ├─── Success
                        │   └─── Play and learn with loaded model
                        │
                        └─── Fail
                            └─── Start new training
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # dqn_agent
    main_model = SnakeLinerDQN(15, 3).to(device=device)
    target_model = SnakeLinerDQN(15, 3).to(device=device)
    dqn_agent = DQNAgent(main_model, target_model, device=device)
    while True:
        user_input = input("Choose mode: (a) Play or (b) Learn and Play (a/b): ").lower()
        if user_input in ['a', 'play']:
            print("Play mode selected.")
            if play_mode(dqn_agent):
                break
        elif user_input in ['b', 'learn and play']:
            # create a callback function to update the figure dynamically
            update_plot = create_training_plotter()
            
            print("Learn and Play mode selected.")
            learn_and_play_mode(dqn_agent, update_plot)
            break
        else:
            print("Invalid input. Please enter 'a' for Play or 'b' for Learn and Play.")

def play_mode(dqn_agent: DQNAgent):
    """
    Return True if agent start playing else False.
    """
    if load_model(dqn_agent):
        print("Previous model loaded successfully.")
        play_with_agent(dqn_agent, 
                        playing_rounds_per_display=playing_rounds_per_display, 
                        total_rounds=playing_total_rounds)
    else:
        prompt = "Without loaded weights, the agent's performance will be poor. Continue? (y/n): "
        if confirm_action(prompt):
            print("Continuing with untrained agent...")
            play_with_agent(dqn_agent, 
                            playing_rounds_per_display=playing_rounds_per_display, 
                            total_rounds=playing_total_rounds)
        else:
            print("Returning to mode selection...")
            return False
    return True

def learn_and_play_mode(dqn_agent: DQNAgent, update_plot):
    if confirm_action("Start training from scratch? (y/n): "):
        print("Starting training from scratch...")
    else:
        print("Attempting to load previous model...")
        if dqn_agent.main_model.load() and dqn_agent.target_model.load():
            print("Previous model loaded successfully.")
        else:
            print("Failed to load previous model. Starting from scratch...")
    
    play_and_learn_with_learning_agent(dqn_agent, 
                                       total_episodes=learning_total_episodes, 
                                       update_plot_callback=update_plot, 
                                       learning_episodes_per_display=learning_episodes_per_display)

def load_model(dqn_agent: DQNAgent):
    """
    Return True if loaded successfully else False.
    """
    if dqn_agent.load():
        print("Previous model loaded successfully.")
        return True
    else:
        print("Failed to load previous model.")
        return False

def confirm_action(prompt: str):
    """
    If player input 'y' then return True, if input 'n' then return False, else loop.
    """
    while True:
        user_input = input(prompt).lower()
        if user_input in ['y', 'n']:
            return user_input == 'y'
        print("Invalid input. Please enter 'y' or 'n'.")

if __name__ == "__main__":
    main()

