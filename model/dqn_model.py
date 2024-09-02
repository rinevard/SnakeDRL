import os
import torch
import torch.nn as nn
import torch.nn.init as init

from common.settings import *

class SnakeLinerDQN(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, dqn_hidden_layer_size), 
            nn.ReLU(), 
            nn.Linear(dqn_hidden_layer_size, dqn_hidden_layer_size), 
            nn.ReLU(), 
            nn.Linear(dqn_hidden_layer_size, dqn_hidden_layer_size), 
            nn.ReLU(), 
            nn.Linear(dqn_hidden_layer_size, output_size)
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.fc(x)
        return x
    
    def save(self, filename='linear_dqn_model.pth'):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        weights_dir = os.path.join(script_dir, 'weights')
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)
        
        full_path = os.path.join(weights_dir, filename)
        
        torch.save(self.state_dict(), full_path)
        print(f"Model saved to {full_path}")

    def load(self, filename='linear_dqn_model.pth'):
        script_dir = os.path.dirname(os.path.abspath(__file__))

        full_path = os.path.join(script_dir, 'weights', filename)

        if not os.path.exists(full_path):
            print(f"Model file not found at {full_path}")
            return False

        try:
            self.load_state_dict(torch.load(full_path, weights_only=True))
            self.eval()
            print(f"Model loaded from {full_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)

