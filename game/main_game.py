from common.settings import *
from common.utils import *
from game.game_display import GameDisplay
from game.game_logic import GameLogic
from game.states import *

class Game:
    def __init__(self, display_on=False):
        # use WIDTH - BLOCK_SIZE to change bottom-right to top-left
        grid_size = convert_screen_coord_to_grid((GAME_WIDTH - BLOCK_SIZE, GAME_HEIGHT - BLOCK_SIZE), BLOCK_SIZE)
        self.logic = GameLogic(grid_size[0], grid_size[1])
        self.display_on = display_on
        self.display = GameDisplay(width=GAME_WIDTH, height=GAME_HEIGHT, block_size=BLOCK_SIZE)
        self.reset()

    def step(self, action: Action) -> None:
        """
        Execute a step in the environment based on the given action.
        """
        self.logic.step(action)
        self.display.handle_event() # make it possible to drag pygame window around
        if self.display_on:
            self._render_and_delay()
        return
    
    def get_state(self) -> State:
        """
        Return a deep copy of current game state.
        """
        return self.logic.get_state().copy()

    def reset(self) -> None:
        return self.logic.reset()

    def set_display_on(self) -> None:
        self.display_on = True
        return 

    def set_display_off(self) -> None:
        self.display_on = False
        return 

    def _close(self) -> None:
        if self.display:
            self.display.close()

    def _render_and_delay(self) -> None:
        """
        Note: Code about time delay is in 'self.display.render'
        """
        cur_state = self.get_state()
        snake = [convert_grid_coord_to_screen(grid_coord, BLOCK_SIZE)
                 for grid_coord in cur_state.snake]
        food = convert_grid_coord_to_screen(cur_state.food, BLOCK_SIZE)
        score = cur_state.get_score()
        game_over = cur_state.is_game_over()
        self.display.render_and_delay(snake, food, score, game_over)
