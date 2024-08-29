from .base_agent import Agent

class GreedyAgent(Agent):
    def __init__(self) -> None:
        super().__init__()

    def get_action(self, state=None) -> int:
        if not state:
            return 0
        
        snake = state['snake']
        food = state['food']
        direction = state['direction']
        head = snake[0]

        # Move horizontally until the x-coordinate aligns with the food, then move vertically
        if direction == (0, -1):
            relative_vec = (food[0] - head[0], food[1] - head[1])
        elif direction == (1, 0):
            relative_vec = (food[1] - head[1], -(food[0] - head[0]))
        elif direction == (0, 1):
            relative_vec = (-(food[0] - head[0]), -(food[1] - head[1]))
        elif direction == (-1, 0):
            relative_vec = (-(food[1] - head[1]), food[0] - head[0])

        if relative_vec[0] < 0:
            return -1
        elif relative_vec[0] > 0:
            return 1
        elif relative_vec[1] > 0 :
            return 1
        else:
            return 0