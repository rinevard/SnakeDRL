import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
import os

class SnakeDQN(nn.Module):
    def __init__(self, input_height: int, input_width: int, in_channels=3, num_actions=3):
        """
        Deep Q-Network for the Snake game.

        Computes Q-values for all possible actions given a state of the game.
        
        Input:
            A tensor with size('batch_size', 'in_channels', 'input_height', 'input_width') representing the current state of the Snake game.
        
        Output:
            A tensor with size ('batch_size', 'num_actions') of Q-values for each possible action (turn right, go straight, turn left).
        """
        super().__init__()
        inner_channel = 16
        output_channel = 4
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, inner_channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(inner_channel, output_channel, kernel_size=3, padding=1),
            nn.ReLU()
        )

        ouput_width = input_width
        output_height = input_height

        self.fc = nn.Linear(ouput_width * output_height * output_channel, num_actions)
        self._initialize_weights()

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
    

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)
    

"""
import time

input_width, input_height = 32, 24
model = SnakeDQN(input_width, input_height)
model.eval()

# random input
batch_size = 64
x = torch.randn(batch_size, 3, input_height, input_width)

for _ in range(10):
    model(x)

# average on 100 tests
num_tests = 100
start_time = time.time()
for _ in range(num_tests):
    with torch.no_grad():
        model(x)
end_time = time.time()

avg_time = (end_time - start_time) / num_tests
print(f"Average predict time: {avg_time*1000:.2f} ms")
"""