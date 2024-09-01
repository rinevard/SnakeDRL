import torch
from enum import Enum

WIDTH = 200
HEIGHT = 200
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
                 grid_height=(HEIGHT - BLOCK_SIZE) // BLOCK_SIZE) -> None:
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
    
    def get_state_tensor(self) -> torch.Tensor:
        """
        Return a tensor representation of the current game state

        Returns:
        A tensor with shape (9,)

        idx:
        direction: 0, 1
        food_relative_pos: 2, 3
        straight_danger_dis: 4
        left_danger_dis: 5
        right_danger_dis: 6
        snake_len: 1
        steps_of_do_nothing: 1

        """
        head: tuple[int, int] = self.get_snake_head()
        food: tuple[int, int] = self.get_food()

        direction: Direction = self.get_direction()
        direction_tuple: tuple[int, int] = convert_direction_to_tuple(direction)
        food_relative_pos: tuple[int, int] = convert_global_pos_to_relative_pos(head, 
                                                               direction, 
                                                               food)
        straight_direction: Direction = self.direction
        left_direction: Direction = change_direction(straight_direction, Action.TURN_LEFT)
        right_direction: Direction = change_direction(straight_direction, Action.TURN_RIGHT)
        straight_danger_dis: int = self.get_closest_danger_dis(head, straight_direction)
        left_danger_dis: int = self.get_closest_danger_dis(head, left_direction)
        right_danger_dis: int = self.get_closest_danger_dis(head, right_direction)
        snake_length: int = len(self.get_snake())
        steps_of_do_nothing: int = self.steps_of_do_nothing

        # shape: (9,)
        # dtype: torch.float32
        return torch.tensor([
            direction_tuple[0], 
            direction_tuple[1], 
            food_relative_pos[0], 
            food_relative_pos[1], 
            straight_danger_dis, 
            left_danger_dis, 
            right_danger_dis, 
            snake_length, 
            steps_of_do_nothing
        ]).to(dtype=torch.float32)

    # def get_state_tensor(self) -> torch.Tensor:
    #     """
    #     Return a tensor representation of the current game state.

    #     Returns:
    #     A tensor with shape (3, grid_height + 1, grid_width + 1) and dtype float32:
        
    #     1. Grid Game Channel:
    #     - 1 represents the snake's body
    #     - 2 represents the snake's head 
    #     - 3 represents food
    #     - 0 represents background (empty cells)
            
    #     2. Direction Channel:
    #     Filled with a single value representing the current direction:
    #     - 0 for Direction.UP
    #     - 1 for Direction.DOWN 
    #     - 2 for Direction.RIGHT
    #     - 3 for Direction.LEFT
            
    #     3. Game Over Channel:
    #     Filled with a single value indicating the game state:
    #     - steps_of_do_nothing if the game is ongoing
    #     - -1 if the game is over

    #     Note:
    #     The tensor uses grid coordinates, where (0, 0) is the top-left corner.
    #     """
    #     shape = (self.grid_height + 1, self.grid_width + 1)
    #     # grid_channel
    #     grid_channel = torch.zeros(shape, dtype=torch.float32)
    #     head = self.get_snake_head()
    #     snake_body = self.get_snake_body()
    #     food = self.food
    #     for seg in snake_body:
    #         grid_channel[(seg[1], seg[0])] = 1.0
    #     grid_channel[head[1], head[0]] = 2.0
    #     grid_channel[food[1], food[0]] = 3.0
    #     # test
    #     grid_channel += 1

    #     # direction_channel
    #     direction = 0
    #     match self.direction:
    #         case Direction.UP:
    #             direction = 0
    #         case Direction.DOWN:
    #             direction = 1
    #         case Direction.RIGHT:
    #             direction = 2
    #         case Direction.LEFT:
    #             direction = 3
    #     direction_channel = torch.full(shape, direction, dtype=torch.float32)

    #     # game_over_channel 
    #     game_over = -1 if self.game_over else self.steps_of_do_nothing
    #     game_over_channel = torch.full(shape, game_over, dtype=torch.float32)

    #     return torch.stack((grid_channel, direction_channel, game_over_channel), dim=0)

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




def convert_direction_to_tuple(direction: Direction) -> tuple[int, int]:
    """
    UP -> (0, -1)
    DOWN -> (0, 1)
    RIGHT -> (1, 0)
    LEFT -> (-1, 0)
    Raises:
        ValueError: If an invalid direction is provided.
    """
    match direction:
        case Direction.UP:
            return (0, -1)
        case Direction.DOWN:
            return (0, 1)
        case Direction.RIGHT:
            return (1, 0)
        case Direction.LEFT:
            return (-1, 0)
        case _:
            raise ValueError(f"Invalid direction: {direction}")

def convert_grid_coord_to_screen(grid_coord: tuple[int, int], block_size: int) -> tuple[int, int]:
    """
    Convert grid coordinates to screen coordinates.

    Args:
        grid_coord: Grid coordinates (x, y).
        block_size: Size of each grid block in pixels.

    Returns:
        Screen coordinates (x, y).

    Note: 
        All coordinates are top-left.
    """
    grid_x, grid_y = grid_coord
    return (grid_x * block_size, grid_y * block_size)

def convert_screen_coord_to_grid(screen_coord: tuple[int, int], block_size: int) -> tuple[int, int]:
    """
    Convert screen coordinates to grid coordinates.

    Args:
        screen_coord: Screen coordinates (x, y).
        block_size: Size of each grid block in pixels.

    Returns:
        Grid coordinates (x, y).

    Note: 
        All coordinates are top-left.
    """
    screen_x, screen_y = screen_coord
    return (screen_x // block_size, screen_y // block_size)

def convert_global_pos_to_relative_pos(origin_point_pos: tuple[int, int], 
                               direction: Direction, 
                               target_global_pos: tuple[int, int]) -> tuple[int, int]:
    """
    Calculate the relative coordinates where the origin is the 'origin_point_pos',
    the y-axis is 'direction', and x-axis is 90 degrees counterclockwise from the 'direction'.
    """
    match direction:
        case Direction.UP:
            relative_pos = (-(target_global_pos[0] - origin_point_pos[0]), 
                            -(target_global_pos[1] - origin_point_pos[1]))
        case Direction.DOWN:
            relative_pos = (target_global_pos[0] - origin_point_pos[0], 
                            target_global_pos[1] - origin_point_pos[1])
        case Direction.RIGHT:
            relative_pos = (-(target_global_pos[1] - origin_point_pos[1]), 
                            target_global_pos[0] - origin_point_pos[0])
        case Direction.LEFT:
            relative_pos = (target_global_pos[1] - origin_point_pos[1], 
                            -(target_global_pos[0] - origin_point_pos[0]))

    return relative_pos

def convert_action_to_action_idx(action: Action) -> int:
    """
    Returns:
        - Action.TURN_RIGHT -> 0 
        - Action.GO_STRAIGHT -> 1
        - Action.TURN_LEFT -> 2
    """
    action_idx = 0
    match action:
        case Action.TURN_RIGHT:
            action_idx = 0
        case Action.GO_STRAIGHT:
            action_idx = 1
        case Action.TURN_LEFT:
            action_idx = 2
    return action_idx

def change_direction(direction: Direction, action: Action) -> Direction:
    directions = [Direction.UP, Direction.LEFT, Direction.DOWN, Direction.RIGHT]
    direction_idx = directions.index(direction)
    new_direction = direction
    if action == Action.TURN_LEFT:  # Turn left
        new_direction = directions[(direction_idx + 1) % len(directions)]
    elif action == Action.TURN_RIGHT:  # Turn right
        new_direction = directions[(direction_idx - 1) % len(directions)]
    return new_direction
























"""
def test_state():
    # 创建4x4的网格
    grid_width = 3  # 索引从0开始，所以最大索引是3
    grid_height = 3

    # 蛇的位置：头部在(1,2)，身体在(1,3)
    snake = [(1,2), (1,3)]
    
    # 方向向上
    direction = Direction.UP
    
    # 食物位置在(0,1)
    food = (0,1)
    
    # 分数为0，游戏未结束
    score = 0
    game_over = False

    # 创建State对象
    state = State(snake, direction, food, score, game_over, grid_width, grid_height)

    # 计算预期的state tensor
    expected_tensor = torch.tensor([0, -1,  1,  1,  3,  2,  3,  2,  0], 
                                   dtype=torch.float32)

    # 获取实际的state tensor
    actual_tensor = state.get_state_tensor()

    # 比较预期和实际的tensor
    if torch.all(torch.eq(expected_tensor, actual_tensor)):
        print("Test passed: The state tensor matches the expected values.")
    else:
        print("Test failed: The state tensor does not match the expected values.")
        print("Expected:", expected_tensor)
        print("Actual:", actual_tensor)

    # 额外测试其他方法
    assert state.get_snake() == snake, "Snake position mismatch"
    assert state.get_snake_head() == (1,2), "Snake head position mismatch"
    assert state.get_snake_body() == [(1,3)], "Snake body mismatch"
    assert state.get_direction() == Direction.UP, "Direction mismatch"
    assert state.get_food() == (0,1), "Food position mismatch"
    assert state.get_score() == 0, "Score mismatch"
    assert not state.is_game_over(), "Game over status mismatch"
    assert state.get_snake_length() == 2, "Snake length mismatch"

    print("All additional tests passed.")

# 运行测试
test_state()
"""