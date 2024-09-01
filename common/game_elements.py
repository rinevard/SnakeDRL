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