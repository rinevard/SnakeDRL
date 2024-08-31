import torch
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
                 game_over: bool, 
                 grid_width=(WIDTH - BLOCK_SIZE) // BLOCK_SIZE, 
                 grid_height=(WIDTH - BLOCK_SIZE) // BLOCK_SIZE) -> None:
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

    def get_state_tensor(self) -> torch.Tensor:
        """
        Return a tensor representation of the current game state.

        Returns:
        A tensor with shape (3, grid_height + 1, grid_width + 1) and dtype float32:
        
        1. Grid Game Channel:
        - 1 represents the snake's head 
        - -1 represents the snake's body
        - 5 represents food
        - 0 represents background (empty cells)
            
        2. Direction Channel:
        Filled with a single value representing the current direction:
        - 0 for Direction.UP
        - 1 for Direction.DOWN 
        - 2 for Direction.RIGHT
        - 3 for Direction.LEFT
            
        3. Game Over Channel:
        Filled with a single value indicating the game state:
        - 0 if the game is ongoing
        - 1 if the game is over

        Note:
        The tensor uses grid coordinates, where (0, 0) is the top-left corner.
        """
        shape = (self.grid_height + 1, self.grid_width + 1)
        # grid_channel
        grid_channel = torch.zeros(shape, dtype=torch.float32)
        head = self.get_snake_head()
        snake_body = self.get_snake_body()
        food = self.food
        for seg in snake_body:
            grid_channel[(seg[1], seg[0])] = -1.0
        grid_channel[head[1], head[0]] = 1.0
        grid_channel[food[1], food[0]] = 5.0

        # direction_channel
        direction = 0
        match self.direction:
            case Direction.UP:
                direction = 0
            case Direction.DOWN:
                direction = 1
            case Direction.RIGHT:
                direction = 2
            case Direction.LEFT:
                direction = 3
        direction_channel = torch.full(shape, direction, dtype=torch.float32)

        # game_over_channel 
        game_over = 1 if self.game_over else 0
        game_over_channel = torch.full(shape, game_over, dtype=torch.float32)

        return torch.stack((grid_channel, direction_channel, game_over_channel), dim=0)
    

"""
def test_get_state_tensor():
    snake = [(1, 2), (1, 1), (1, 0)]
    direction = Direction.DOWN  
    food = (3, 3)  
    score = 2
    game_over = False
    
    # create a 4*4 state
    state = State(snake, direction, food, score, game_over, grid_width=3, grid_height=3)
    
    tensor = state.get_state_tensor()
    
    # print shape
    print("Tensor shape:", tensor.shape)
    
    # print channels
    print("\nGrid Channel:")
    print(tensor[0])
    
    print("\nDirection Channel:")
    print(tensor[1])

    print("\nGame Over Channel:")
    print(tensor[2])
    
    assert tensor[0, 2, 1] == 1.0, "Snake head should be 1.0"
    assert tensor[0, 1, 1] == -1.0, "Snake body should be -1.0"
    assert tensor[0, 0, 1] == -1.0, "Snake body should be -1.0"
    assert tensor[0, 3, 3] == 5.0, "Food should be 5.0"
    assert torch.all(tensor[1] == 1), "Direction should be 1 (DOWN)"
    assert torch.all(tensor[2] == 0), "Game over should be 0 (False)"
    
    print("\nAll assertions passed. The get_state_tensor method seems to be working correctly.")
"""
