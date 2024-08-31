from game.main_game import Game
from agent.greedy_agent import GreedyAgent
from agent.base_agent import Agent
from common.helper import convert_screen_coord_to_grid

WIDTH = 640
HEIGHT = 480
BLOCK_SIZE = 20

def play_with_agent(agent: Agent):
    game = Game(display_on=False)
    game_over_times = 0
    while True:
        if game_over_times % 100 == 0:
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

greedy_agent = GreedyAgent()
play_with_agent(greedy_agent)
