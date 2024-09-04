from common.settings import *
from common.utils import *

class State:
    def __init__(self, snake: list[tuple[int, int]], 
                 direction: Direction, 
                 food: tuple[int, int], 
                 score: int, 
                 game_over: bool, 
                 grid_width=(GAME_WIDTH - BLOCK_SIZE) // BLOCK_SIZE, 
                 grid_height=(GAME_HEIGHT - BLOCK_SIZE) // BLOCK_SIZE) -> None:
        """
        Parameters:
            snake: List of coordinates representing the snake's body.
                Each element is a tuple. snake[0] is the position of the snake's head.
            direction: Current direction of the snake.
            food: Coordinates of the food as a tuple.
            score: Current score of the game.
            game_over: Boolean indicating if the game is over.
            grid_width: x coordinate is in [0, grid_width]
            grid_height: y coodinate is in [0, grid_height]

        Note:
            Coordinates represent the top-left corner of each cell.
            Coordinates are grid coordinates.
        """
        self.snake = snake
        self.direction = direction
        self.food = food
        self.score = score
        self.game_over = game_over
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.steps_of_do_nothing = 0

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
    
    def get_grid_width(self) -> int:
        return self.grid_width
    
    def get_grid_height(self) -> int:
        return self.grid_height
    
    def get_closest_danger_dis(self, pos: tuple[int, int], direction: Direction) -> int:
        """
        Return the closest danger distance in 'direction' from 'pos'
        """
        direction = convert_direction_to_tuple(direction)
        for step in range(max(self.grid_width, self.grid_height) + 2):
            new_pos = (pos[0] + step * direction[0], pos[1] + step * direction[1])
            if (new_pos in self.get_snake_body() or 
                new_pos[0] < 0 or new_pos[0] > self.grid_width or
                new_pos[1] < 0 or new_pos[1] > self.grid_height):
                return step
        return max(self.grid_width, self.grid_height) + 2
    
    def get_closest_danger_distances(self, pos: tuple[int, int], straight_direction: Direction) -> tuple[int, int, int]:
        """
        return (right_danger_dis, straight_danger_dis, left_danger_dis)
        """
        left_direction: Direction = change_direction(straight_direction, Action.TURN_LEFT)
        right_direction: Direction = change_direction(straight_direction, Action.TURN_RIGHT)

        right_danger_dis: int = self.get_closest_danger_dis(pos, right_direction)
        straight_danger_dis: int = self.get_closest_danger_dis(pos, straight_direction)
        left_danger_dis: int = self.get_closest_danger_dis(pos, left_direction)

        return (right_danger_dis, straight_danger_dis, left_danger_dis)

    def get_grid_state(self) -> torch.Tensor:
        """
        Return a tensor where:
        - 1 represents the snake's body
        - 2 represents the snake's head 
        - 3 represents food
        - 0 represents background (empty cells)
        """
        shape = (self.grid_height + 1, self.grid_width + 1)
        # grid_tensor
        grid_tensor = torch.zeros(shape, dtype=torch.float32)
        head = self.get_snake_head()
        snake_body = self.get_snake_body()
        food = self.food
        for seg in snake_body:
            grid_tensor[(seg[1], seg[0])] = 1.0
        grid_tensor[head[1], head[0]] = 2.0
        grid_tensor[food[1], food[0]] = 3.0

        return grid_tensor

    def copy(self):
        """
        Create a deep copy of the current State object.
        """
        return State(
            snake=self.snake[:],
            direction=self.direction,
            food=self.food,
            score=self.score,
            game_over=self.game_over,
            grid_width=self.grid_width,
            grid_height=self.grid_height
        )
