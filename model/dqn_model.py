import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
import os

class SnakeDQN(nn.Module):
    def __init__(self, input_width: int, input_height: int, in_channels=1, num_actions=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        output_width_1 = (input_width - 3 + 2 * 1) // 2 + 1
        output_height_1 = (input_height - 3 + 2 * 1) // 2 + 1
        output_width_2 = (output_width_1 - 3 + 2 * 1) // 2 + 1
        output_height_2 = (output_height_1 - 3 + 2 * 1) // 2 + 1

        self.fc = nn.Linear(output_width_2 * output_height_2, num_actions)
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
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                init.constant_(m.bias, 0)
    


"""
import time

input_width, input_height = 32, 24
model = SnakeDQN(input_width, input_height)

# random input
x = torch.randn(1, 1, input_height, input_width)

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