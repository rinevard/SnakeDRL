from game.main_game import Game
from agent.greedy_agent import GreedyAgent
from agent.base_agent import Agent

def play_with_agent(agent: Agent):
    new_game = Game(need_display=True)
    game_over_times = 0
    while True:
        if game_over_times % 100 == 0:
            new_game.set_display_on()
        else:
            new_game.set_display_off()
        game_state_dic = new_game.get_state()

        action = agent.get_action(game_state_dic)
        new_game.step(action)

        if game_state_dic['is_game_over']:
            game_over_times += 1
            print(f"Times: {game_over_times}, Score: {game_state_dic['score']}")
            new_game.reset()


greedy_agent = GreedyAgent()
play_with_agent(greedy_agent)