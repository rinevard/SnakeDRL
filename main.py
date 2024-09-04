import torch

from common.settings import *
from agent.play_game_with_agent import *
from agent.mlp_agent import MLPAgent
from agent.cnn_agent import CNNAgent

def main():
    """
    Start
    │
    └─── Choose agent type
        │
        ├─── MLP Agent
        │
        └─── CNN Agent
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
                                    │
                                    └─── Continue without loaded model?
                                        │
                                        ├─── Yes
                                        │   └─── Play and learn with untrained model
                                        │
                                        └─── No
                                            └─── Return to mode selection
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    while True:
        agent_type = input("Choose agent type: (a) MLP Agent or (b) CNN Agent (a/b): ").lower()
        if agent_type in ['a', 'mlp']:
            agent = MLPAgent(device=device)
            print("MLP Agent selected.")
            break
        elif agent_type in ['b', 'cnn']:
            agent = CNNAgent(device=device)
            print("CNN Agent selected.")
            break
        else:
            print("Invalid input. Please enter 'a' for MLP Agent or 'b' for CNN Agent.")
    
    while True:
        user_input = input("Choose mode: (a) Play or (b) Learn and Play (a/b): ").lower()
        if user_input in ['a', 'play']:
            print("Play mode selected.")
            if play_mode(agent):
                break
        elif user_input in ['b', 'learn and play']:
            # create a callback function to update the figure dynamically
            update_plot = create_training_plotter()
            
            print("Learn and Play mode selected.")
            if learn_and_play_mode(agent, update_plot):
                break
        else:
            print("Invalid input. Please enter 'a' for Play or 'b' for Learn and Play.")

def play_mode(dqn_agent: LearningAgent):
    """
    Return True if agent starts playing else False.
    """
    if load_model(dqn_agent):
        print("Previous model loaded successfully.")
        play_with_agent(dqn_agent, 
                        playing_rounds_per_display=playing_rounds_per_display, 
                        total_rounds=playing_total_rounds)
        return True
    else:
        prompt = "Without loaded weights, the agent's performance will be poor. Continue? (y/n): "
        if confirm_action(prompt):
            print("Continuing with untrained agent...")
            play_with_agent(dqn_agent, 
                            playing_rounds_per_display=playing_rounds_per_display, 
                            total_rounds=playing_total_rounds)
            return True
        else:
            print("Returning to mode selection...")
            return False

def learn_and_play_mode(dqn_agent: LearningAgent, update_plot):
    """
    Return True if agent starts learning and playing else False.
    """
    if not confirm_action("Start training from scratch? (y/n): "):
        print("Attempting to load previous model...")
        if load_model(dqn_agent):
            print("Previous model loaded successfully.")
        else:
            print("Failed to load previous model.")
            if not confirm_action("Continue with untrained model? (y/n): "):
                print("Returning to mode selection...")
                return False
    
    print("Starting training...")
    play_and_learn_with_learning_agent(dqn_agent, 
                                       total_episodes=learning_total_episodes, 
                                       update_plot_callback=update_plot, 
                                       learning_episodes_per_display=learning_episodes_per_display)
    return True

def load_model(dqn_agent: LearningAgent):
    """
    Load both main and target models.
    Return True if loaded successfully else False.
    """
    if dqn_agent.load():
        return True
    else:
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