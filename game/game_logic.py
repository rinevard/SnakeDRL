import random
from common.helper import *

class GameLogic:
    def __init__(self, grid_width: int, grid_height: int) -> None:
        """
        The 'shape' of grid-world will be (grid_width + 1, grid_height + 1), 
        x in [0, grid_width], y in [0, grid_height]
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.game_state: State = None
        self.reset()

    def _place_food(self) -> None:
        while True:
            x = random.randint(0, self.grid_width)
            y = random.randint(0, self.grid_height)
            if (x, y) not in self.game_state.snake:
                self.game_state.food = (x, y)
                break

    def step(self, action: Action) -> None:
        """
        Execute a step in the environment based on the given action.
        """
        # 0. if game is over, return
        if self.game_state.game_over:
            return

        # 1. Update direction based on action
        self.game_state.direction = change_direction(self.game_state.direction, action)
        direction = self.game_state.direction

        # 2. Move snake
        snake = self.game_state.snake
        direction_tuple = convert_direction_to_tuple(direction)
        new_head = (
            snake[0][0] + direction_tuple[0],
            snake[0][1] + direction_tuple[1]
        )

        # 3. Check if snake hit the wall
        if (new_head[0] < 0 or new_head[0] > self.grid_width or
            new_head[1] < 0 or new_head[1] > self.grid_height or
            self.game_state.steps_of_do_nothing >= 4 * (WIDTH // BLOCK_SIZE) * len(self.game_state.snake)):
            self.game_state.game_over = True
            return

        # 4. Move snake
        self.game_state.snake.insert(0, new_head)

        # 5. Check if snake ate food
        if new_head == self.game_state.food:
            self.game_state.steps_of_do_nothing = 0
            self.game_state.score += 1
            self._place_food()
        else:
            self.game_state.steps_of_do_nothing += 1
            self.game_state.snake.pop()

        # 6. Check if snake collided with itself
        if new_head in self.game_state.snake[1:]:
            self.game_state.game_over = True

    def get_state(self) -> State:
        return self.game_state
    
    def reset(self) -> None:
        center_x = self.grid_width // 2
        center_y = self.grid_height // 2
        snake = [(center_x, center_y), (center_x-1, center_y), (center_x-2, center_y)]
        direction = Direction.RIGHT  # default direction is right
        score = 0
        game_over = False

        self.game_state = State(snake, direction, None, score, game_over)
        self.game_state.steps_of_do_nothing = 0
        self._place_food()






















# test
class TestGameLogic:
    def __init__(self, grid_width: int, grid_height: int) -> None:
        """
        The 'shape' of grid-world will be (grid_width + 1, grid_height + 1), 
        x in [0, grid_width], y in [0, grid_height]
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.game_state: State = None
        self.reset()

    def _place_food(self) -> None:
        while True:
            x = random.randint(0, self.grid_width)
            y = random.randint(0, self.grid_height)
            if (x, y) not in self.game_state.snake:
                self.game_state.food = (x, y)
                break

    def step(self, action: Action) -> None:
        """
        Execute a step in the environment based on the given action.
        """
        # 0. if game is over, return
        if self.game_state.game_over:
            return

        # 1. Update direction based on action
        directions = [Direction.UP, Direction.LEFT, Direction.DOWN, Direction.RIGHT]
        direction = self.game_state.direction
        direction_idx = directions.index(direction)
        if action == Action.TURN_LEFT:  # Turn left
            self.game_state.direction = directions[(direction_idx + 1) % len(directions)]
        elif action == Action.TURN_RIGHT:  # Turn right
            self.game_state.direction = directions[(direction_idx - 1) % len(directions)]
        direction = self.game_state.direction

        # 2. Move snake
        snake = self.game_state.snake
        direction_tuple = convert_direction_to_tuple(direction)
        new_head = (
            (snake[0][0] + direction_tuple[0]) % (self.grid_width + 1),
            (snake[0][1] + direction_tuple[1]) % (self.grid_height + 1)
        )

        # 3. Check if snake has been doing nothing for too long
        if self.game_state.steps_of_do_nothing >= 4 * (WIDTH // BLOCK_SIZE) * len(self.game_state.snake):
            self.game_state.game_over = True
            return

        # 4. Move snake
        self.game_state.snake.insert(0, new_head)

        # 5. Check if snake ate food
        if new_head == self.game_state.food:
            self.game_state.steps_of_do_nothing = 0
            self.game_state.score += 1
            self._place_food()
        else:
            self.game_state.steps_of_do_nothing += 1
        # do not grow
        self.game_state.snake.pop()


    def get_state(self) -> State:
        return self.game_state
    
    def reset(self) -> None:
        center_x = self.grid_width // 2
        center_y = self.grid_height // 2
        snake = [(center_x, center_y), (center_x-1, center_y), (center_x-2, center_y)]
        direction = Direction.RIGHT  # default direction is right
        score = 0
        game_over = False

        self.game_state = State(snake, direction, None, score, game_over)
        self.game_state.steps_of_do_nothing = 0
        self._place_food()