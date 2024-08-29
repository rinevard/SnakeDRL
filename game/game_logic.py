import random
from common.game_elements import Action

class GameLogic:
    def __init__(self, grid_width, grid_height) -> None:
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.reset()

    def _place_food(self) -> None:
        while True:
            x = random.randint(0, self.grid_width)
            y = random.randint(0, self.grid_height)
            if (x, y) not in self.snake:
                self.food = (x, y)
                break

    def step(self, action: Action) -> None:
        """
        Execute a step in the environment based on the given action.
        """
        # 0. if game is over, return
        if self.is_game_over:
            return

        # 1. Update direction based on action
        if action == Action.TURN_LEFT:  # Turn left
            self.direction = (self.direction[1], -self.direction[0])
        elif action == Action.TURN_RIGHT:  # Turn right
            self.direction = (-self.direction[1], self.direction[0])

        # 2. Move snake
        new_head = (
            self.snake[0][0] + self.direction[0],
            self.snake[0][1] + self.direction[1]
        )

        # 3. Check if snake hit the wall
        if (new_head[0] < 0 or new_head[0] > self.grid_width or
            new_head[1] < 0 or new_head[1] > self.grid_height):
            self.is_game_over = True
            return

        # 4. Move snake
        self.snake.insert(0, new_head)

        # 5. Check if snake ate food
        if new_head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()

        # 6. Check if snake collided with itself
        if new_head in self.snake[1:]:
            self.is_game_over = True

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
            'direction': (0, 1)
            'food': (8, 3),
            'score': 2
            'is_game_over': False
        }
        """
        return {
            'snake': self.snake,
            'direction': self.direction,
            'food': self.food,
            'score': self.score, 
            'is_game_over': self.is_game_over
        }
    
    def reset(self) -> None:
        center_x = self.grid_width // 2
        center_y = self.grid_height // 2
        self.snake = [(center_x, center_y), (center_x-1, center_y), (center_x-2, center_y)]
        self.direction = (1, 0)  # default direction is right
        self.food = None
        self._place_food()
        self.score = 0
        self.is_game_over = False