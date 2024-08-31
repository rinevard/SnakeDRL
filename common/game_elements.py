from enum import Enum

WIDTH = 640
HEIGHT = 480
BLOCK_SIZE = 20

class Action(Enum):
    TURN_RIGHT = 0
    GO_STRAIGHT = 1
    TURN_LEFT = 2

class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    RIGHT = (1, 0)
    LEFT = (-1, 0)

class State:
    def __init__(self, snake: list[tuple[int, int]], 
                 direction: Direction, 
                 food: tuple[int, int], 
                 score: int, 
                 game_over: bool) -> None:
        """
        Parameters:
            snake: List of coordinates representing the snake's body.
                Each element is a tuple. snake[0] is the position of the snake's head.
            direction: Current direction of the snake.
            food: Coordinates of the food as a tuple.
            score: Current score of the game.
            game_over: Boolean indicating if the game is over.

        Note:
            Coordinates represent the top-left corner of each cell.
            Coordinates are grid coordinates.
        """
        self.snake = snake
        self.direction = direction
        self.food = food
        self.score = score
        self.game_over = game_over

    def get_snake(self) -> list[tuple[int, int]]:
        """
        Return the shallow copy of the snake
        """
        return self.snake[:]

    def get_snake_head(self) -> tuple[int, int]:
        """
        Return the coordinates of the snake's head
        """
        return self.snake[0]

    def get_snake_body(self) -> list[tuple[int, int]]:
        """
        Return the coordinates of the snake's body (excluding the head)
        """
        return self.snake[1:]

    def get_direction(self) -> Direction:
        """
        Return the current direction of the snake
        """
        return self.direction

    def get_food(self) -> tuple[int, int]:
        """
        Return the coordinates of the food
        """
        return self.food

    def get_score(self) -> int:
        """
        Return the current score
        """
        return self.score

    def is_game_over(self) -> bool:
        """
        Return whether the game is over
        """
        return self.game_over

    def get_snake_length(self) -> int:
        """
        Return the length of the snake
        """
        return len(self.snake)