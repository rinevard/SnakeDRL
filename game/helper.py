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