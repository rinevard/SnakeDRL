import torch

def grid_coord_to_screen(grid_coord: tuple[int, int], block_size: int) -> tuple[int, int]:
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


def screen_coord_to_grid(screen_coord: tuple[int, int], block_size: int) -> tuple[int, int]:
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

def global_pos_to_relative_pos(origin_point_pos: tuple[int, int], 
                               direction: tuple[int, int], 
                               target_global_pos: tuple[int, int]) -> tuple[int, int]:
    """
    Calculate the relative coordinates where the origin is the 'origin_point_pos',
    the y-axis is 'direction', and x-axis is 90 degrees counterclockwise from the 'direction'.

    Note:
        direction must be one of [(0, -1), (0, 1), (1, 0), (-1, 0)]
    """
    if direction == (0, 1):
        relative_pos = (target_global_pos[0] - origin_point_pos[0], 
                        target_global_pos[1] - origin_point_pos[1])
    elif direction == (0, -1):
        relative_pos = (-(target_global_pos[0] - origin_point_pos[0]), 
                        -(target_global_pos[1] - origin_point_pos[1]))
    elif direction == (1, 0):
        relative_pos = (-(target_global_pos[1] - origin_point_pos[1]), 
                        target_global_pos[0] - origin_point_pos[0])
    elif direction == (-1, 0):
        relative_pos = (target_global_pos[1] - origin_point_pos[1], 
                        -(target_global_pos[0] - origin_point_pos[0]))

    return relative_pos

def state_dic_to_state_tensor(state_dict, width=640, height=480, block_size=20) -> torch.Tensor:
    """
    Parameters:
        state_dict: A dictionary containing the following keys:
        - 'snake' (list of tuple): List of coordinates representing the snake's body.
            Each element is a tuple (x: int, y: int). snake[0] is the position of the snake's head.
        - 'direction' (tuple): x = 1 if right else -1; y = 1 if down else -1.
        - 'food' (tuple): Coordinates of the food as a tuple (x: int, y: int).
        - 'score' (int)
        - 'is_game_over' (bool)

    Returns:
        A tensor of shape ((width + block_size) // block_size, 
                            (height + block_size) // block_size) 
                        representing current game state.
        
    Note:
        Coordinates represent the top-left corner of each cell.
        Coordinates are grid coordinates.
    """
    assert state_dict, "'state_dict' is " + str(state_dict) + " !"

    # first build a game grid, then padding with '-10'
    # possible places: (range(0, grid_w), range(0, grid_h))
    grid_w, grid_h = screen_coord_to_grid((width, height), 
                                          block_size)
    grid = torch.zeros((grid_w, grid_h))

    snake = state_dict['snake']
    direction = state_dict['direction']
    food = state_dict['food']

    head = snake[0] # head
    if head:
        grid[head] += 1
    snake_body = snake[1:]  # body
    if snake_body:
        body_x, body_y = zip(*snake_body)
        grid[list(body_x), list(body_y)] += (-10)
    grid[food] += 10 # food

    # fill the edge    
    padded_grid = torch.full((grid_w + 2, grid_h + 2), fill_value=-10)
    padded_grid[1:-1, 1:-1] = grid
    padded_grid[snake[0][0] + 1 + direction[0], 
                snake[0][1] + 1 + direction[1]] += 2  # direction

    return padded_grid


"""
import time
# 测试代码
def test():
    state_dict = {
        'snake': [(5, 5), (4, 5), (3, 5)],
        'direction': (0, -1),
        'food': (0, 1),
        'score': 2,
        'is_game_over': False
    }
    
    tensor = state_dic_to_state_tensor(state_dict, width=640, height=480, block_size=20)

num_tests = 100
start_time = time.time()
for _ in range(num_tests):
    test()
end_time = time.time()

avg_time = (end_time - start_time) / num_tests
print(f"Average predict time: {avg_time*1000:.2f} ms")
"""