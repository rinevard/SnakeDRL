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
        ├─── (a) MLP Agent
        │
        └─── (b) CNN Agent
            │
            └─── Choose mode
                │
                ├─── (a) Play
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
                │               ├─── (a) Return to mode selection
                │               │
                │               └─── (b) Play with untrained model
                │
                └─── (b) Learn and Play
                    │
                    └─── Start training from scratch?
                        │
                        ├─── (a) Load previous model
                        │   │
                        │   ├─── Success
                        │   │   └─── Play and learn with loaded model
                        │   │
                        │   └─── Fail
                        │       │
                        │       └─── Continue without loaded model?
                        │           │
                        │           ├─── (a) Return to mode selection 
                        │           │
                        │           └─── (b) Training from scratch
                        │
                        └─── (b) Start new training
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if select_action("Choose agent type:\n(a) MLP Agent\n(b) CNN Agent"):
        agent = MLPAgent(device=device)
        print("MLP Agent selected.")
    else:
        agent = CNNAgent(device=device)
        print("CNN Agent selected.")
    
    while True:
        if select_action("Choose mode:\n(a) Play\n(b) Learn and Play"):
            print("Play mode selected.")
            if play_mode(agent):
                break
        else:
            print("Learn and Play mode selected.")
            if learn_and_play_mode(agent):
                break

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
        print("Failed to load previous model.")
        if select_action("Continue without loaded model?\n(a) Return to mode selection\n(b) Play with untrained model"):
            print("Returning to mode selection...")
            return False
        else:
            print("Playing with untrained model...")
            play_with_agent(dqn_agent, 
                            playing_rounds_per_display=playing_rounds_per_display, 
                            total_rounds=playing_total_rounds)
            return True

def learn_and_play_mode(dqn_agent: LearningAgent):
    """
    Return True if agent starts learning and playing else False.
    """
    if select_action("Start training from scratch?\n(a) Load previous model\n(b) Start new training"):
        print("Attempting to load previous model...")
        if load_model(dqn_agent):
            print("Previous model loaded successfully.")
            print("Starting training with loaded model...")
            update_plot = create_training_plotter()
            play_and_learn_with_learning_agent(dqn_agent, 
                                               total_episodes=learning_total_episodes, 
                                               update_plot_callback=update_plot, 
                                               learning_episodes_per_display=learning_episodes_per_display)
            return True
        else:
            print("Failed to load previous model.")
            if select_action("Continue without loaded model?\n(a) Return to mode selection\n(b) Training from scratch"):
                print("Returning to mode selection...")
                return False
            else:
                print("Playing and learning with untrained model...")
    else:
        print("Starting new training...")
    update_plot = create_training_plotter()
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

def select_action(prompt: str):
    """
    If player input 'a' then return True, if input 'b' then return False, else loop.
    """
    while True:
        user_input = input(prompt + "\n(a/b): ").lower()
        if user_input in ['a', 'b']:
            print("\n")
            return user_input == 'a'
        print("Invalid input. Please enter 'a' or 'b'.")

if __name__ == "__main__":
    main()