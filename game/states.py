from common.game_elements import *
from common.utils import *

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


    def get_state_tensor(self) -> torch.Tensor:
        """
        Return a tensor representation of the current game state

        Returns:
        A tensor with shape (15,)

        idx:
        direction_x: 0
        direction_y: 1
        food_relative_pos: 2, 3
        snake_len: 4
        steps_of_do_nothing: 5

        right_danger_dis_after_turn_right: 6
        straight_danger_dis_after_turn_right: 7
        left_danger_dis_after_turn_right: 8
        
        right_danger_dis_after_go_straight: 9
        straight_danger_dis_after_go_straight: 10
        left_danger_dis_after_go_straight: 11

        right_danger_dis_after_turn_left: 12
        straight_danger_dis_after_turn_left: 13
        left_danger_dis_after_turn_left: 14
        """
        head: tuple[int, int] = self.get_snake_head()
        body: list[tuple[int, int]] = self.get_snake_body()
        food: tuple[int, int] = self.get_food()

        direction: Direction = self.get_direction()
        # direction_x, direction_y
        direction_tuple: tuple[int, int] = convert_direction_to_tuple(direction)
        # food_relative_pos
        food_relative_pos: tuple[int, int] = convert_global_pos_to_relative_pos(head, 
                                                               direction, 
                                                               food)

        # snake_len
        snake_length: int = len(self.get_snake())
        # steps_of_do_nothing
        steps_of_do_nothing: int = self.steps_of_do_nothing

        # danger_distances
        straight_direction: Direction = self.direction
        left_direction: Direction = change_direction(straight_direction, Action.TURN_LEFT)
        right_direction: Direction = change_direction(straight_direction, Action.TURN_RIGHT)

        right_direction_tuple: tuple[int, int] = convert_direction_to_tuple(right_direction)
        straight_direction_tuple: tuple[int, int] = convert_direction_to_tuple(straight_direction)
        left_direction_tuple: tuple[int, int] = convert_direction_to_tuple(left_direction)
        
        head_after_turn_right: tuple[int, int] = (head[0] + right_direction_tuple[0], 
                                                 head[1] + right_direction_tuple[1])
        head_after_go_straight: tuple[int, int] = (head[0] + straight_direction_tuple[0], 
                                                   head[1] + straight_direction_tuple[1])
        head_after_turn_left: tuple[int, int] = (head[0] + left_direction_tuple[0], 
                                                 head[1] + left_direction_tuple[1])
        
        danger_distances_after_turn_right = self.get_closest_danger_distances(head_after_turn_right, 
                                                                              right_direction)
        danger_distances_after_go_straight = self.get_closest_danger_distances(head_after_go_straight, 
                                                                               straight_direction)
        danger_distances_after_turn_left = self.get_closest_danger_distances(head_after_turn_left, 
                                                                             left_direction)
        # shape: (15,)
        # dtype: torch.float32
        return torch.tensor([
            direction_tuple[0], 
            direction_tuple[1], 
            food_relative_pos[0], 
            food_relative_pos[1], 
            snake_length, 
            steps_of_do_nothing, 
            danger_distances_after_turn_right[0], 
            danger_distances_after_turn_right[1], 
            danger_distances_after_turn_right[2],
            danger_distances_after_go_straight[0], 
            danger_distances_after_go_straight[1], 
            danger_distances_after_go_straight[2],
            danger_distances_after_turn_left[0], 
            danger_distances_after_turn_left[1], 
            danger_distances_after_turn_left[2], 
        ]).to(dtype=torch.float32)

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

