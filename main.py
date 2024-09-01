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

    # dqn_agent
    main_model = SnakeLinerDQN(15, 3)
    target_model = SnakeLinerDQN(15, 3)
    dqn_agent = DQNAgent(main_model, target_model)
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

def play_mode(dqn_agent):
    """
    Return True if agent start playing else False.
    """
    if load_model(dqn_agent):
        play_with_agent(dqn_agent)
    else:
        prompt = "Without loaded weights, the agent's performance will be poor. Continue? (y/n): "
        if confirm_action(prompt):
            print("Continuing with untrained agent...")
            play_with_agent(dqn_agent)
        else:
            print("Returning to mode selection...")
            return False
    return True

def learn_and_play_mode(dqn_agent: DQNAgent, update_plot):
    load_successfully = False
    if confirm_action("Start training from scratch? (y/n): "):
        print("Starting training from scratch...")
    else:
        print("Attempting to load previous model...")
        if dqn_agent.main_model.load() and dqn_agent.target_model.load():
            load_successfully = True
            print("Previous model loaded successfully.")
        else:
            print("Failed to load previous model. Starting from scratch...")
    epsilon_start = 0.8
    if load_successfully:
        epsilon_start = 0.2
    play_and_learn_with_dqn_agent(dqn_agent, epsilon_start=epsilon_start, 
                                  update_plot_callback=update_plot)

def load_model(dqn_agent: DQNAgent):
    """
    Return True if loaded successfully else False.
    """
    if dqn_agent.main_model.load():
        print("Previous model loaded successfully.")
        return True
    else:
        print("Failed to load previous model.")
        return False

def confirm_action(prompt):
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

