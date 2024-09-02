import torch
from enum import Enum
from common.settings import *

class Action(Enum):
    TURN_RIGHT = 0
    GO_STRAIGHT = 1
    TURN_LEFT = 2

class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    RIGHT = (1, 0)
    LEFT = (-1, 0)

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
