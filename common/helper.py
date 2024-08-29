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