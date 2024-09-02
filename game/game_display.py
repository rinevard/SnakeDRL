import pygame

from common.settings import *

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
LIGHT_BLUE = (0, 127, 255)
BLUE = (0, 0, 255)

class GameDisplay:
    def __init__(self, width=GAME_WIDTH, height=GAME_HEIGHT, block_size=BLOCK_SIZE):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.block_size = block_size

    def render_and_delay(self, snake: list[tuple[int, int]], food: tuple[int, int], score: int, is_game_over: bool):
        """
        Render the game state on the screen and delay for a bit of time.
        
        Args:
            snake: List of screen coordinates for snake body. snake[0] is the head.
            food: Screen coordinates of the food.
            score: Current game score.
            is_game_over: Boolean indicating if the game is over.
        
        Note:
            All coordinates should be screen coordinates instead of grid coordinates.
        """
        self.screen.fill(BLACK)

        # food
        pygame.draw.rect(self.screen, RED, (food[0], food[1], self.block_size, self.block_size))
        
        # body of snake
        for seg in snake[1:]:
            pygame.draw.rect(self.screen, LIGHT_BLUE, pygame.Rect(seg[0], seg[1], self.block_size, self.block_size))
        
        # head of snake
        head = snake[0]
        pygame.draw.rect(self.screen, BLUE, pygame.Rect(head[0], head[1], self.block_size, self.block_size))
        
        # score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))
        
        # game_over
        if is_game_over:
            font_big = pygame.font.Font(None, 72)
            game_over_text = font_big.render("GAME OVER", True, RED)
            text_rect = game_over_text.get_rect(center=(self.screen.get_width() / 2, self.screen.get_height() / 2))
            self.screen.blit(game_over_text, text_rect)
        
        pygame.display.flip()
        self.clock.tick(GAME_SPEED)  # 控制帧率

    def close(self):
        pygame.display.quit()

    def handle_event(self):
        for event in pygame.event.get():
            pass