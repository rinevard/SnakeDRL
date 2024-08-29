from .game_logic import GameLogic
from .game_display import GameDisplay
from .helper import screen_coord_to_grid
from .helper import grid_coord_to_screen

WIDTH = 640
HEIGHT = 480
BLOCK_SIZE = 20

class Game:
    def __init__(self, display_on=False):
        # use WIDTH - BLOCK_SIZE to change bottom-right to top-left
        grid_size = screen_coord_to_grid((WIDTH - BLOCK_SIZE, HEIGHT - BLOCK_SIZE), BLOCK_SIZE)
        self.logic = GameLogic(grid_size[0], grid_size[1])
        self.display_on = display_on
        self.display = GameDisplay(width=WIDTH, height=HEIGHT, block_size=BLOCK_SIZE) if self.display_on else None

    def step(self, action):
        """
        Execute a step in the environment based on the given action.
        Parameters:
            action (int): The action to be performed.
                - 0: Continue in the current direction
                - -1: Turn left
                - 1: Turn right
        """
        self.logic.step(action)
        self.display.handle_event() # make it possible to move pygame window
        if self.display_on:
            self._render_and_delay()
        return
    
    def get_state(self):
        """
        Get current state of the game.

        Returns:
        dict: A dictionary containing the following keys:
        - 'snake' (list of tuple): List of coordinates representing the snake's body.
          Each element is a tuple (x: int, y: int). snake[0] is the position of the snake's head.
        - 'direction' (tuple): x = 1 if right else -1; y = 1 if down else -1.
        - 'food' (tuple): Coordinates of the food as a tuple (x: int, y: int).
        - 'score' (int)
        - 'is_game_over' (bool)

        Note:
            Coordinates represent the top-left corner of each cell.
            Coordinates are grid coordinates.

        Example:
        {
            'snake': [(5, 5), (4, 5), (3, 5)],
            'direction': (-1, 1)
            'food': (8, 3),
            'score': 2
            'is_game_over': False
        }
        """
        return self.logic.get_state()

    def reset(self):
        return self.logic.reset()

    def set_display_on(self):
        self.display_on = True
        return 

    def set_display_off(self):
        self.display_on = False
        return 

    def _close(self):
        if self.display:
            self.display.close()

    def _render_and_delay(self):
        """
        Note: Code about time delay is in 'self.display.render'
        """
        state_dic = self.logic.get_state()
        snake = [grid_coord_to_screen(grid_coord, BLOCK_SIZE) for grid_coord in state_dic['snake']]
        food = grid_coord_to_screen(state_dic['food'], BLOCK_SIZE)
        score = state_dic['score']
        is_game_over = state_dic['is_game_over']
        self.display.render_and_delay(snake, food, score, is_game_over)


