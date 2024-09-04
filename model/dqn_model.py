import os
import torch
import torch.nn as nn
import torch.nn.init as init

from common.settings import *

class SnakeMLP(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, mlp_hidden_layer_size), 
            nn.ReLU(), 
            nn.Linear(mlp_hidden_layer_size, mlp_hidden_layer_size), 
            nn.ReLU(), 
            nn.Linear(mlp_hidden_layer_size, mlp_hidden_layer_size), 
            nn.ReLU(), 
            nn.Linear(mlp_hidden_layer_size, output_size)
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.fc(x)
        return x
    
    def save(self, filename='mlp_model.pth'):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        weights_dir = os.path.join(script_dir, 'weights')
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)
        
        full_path = os.path.join(weights_dir, filename)
        
        torch.save(self.state_dict(), full_path)
        print(f"Model saved to {full_path}")

    def load(self, filename='mlp_model.pth', device=None):
        script_dir = os.path.dirname(os.path.abspath(__file__))

        full_path = os.path.join(script_dir, 'weights', filename)

        if not os.path.exists(full_path):
            print(f"Model file not found at {full_path}")
            return False

        try:
            self.load_state_dict(torch.load(full_path, weights_only=True, map_location=device))
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

class SnakeCNN(nn.Module):
    def __init__(self, input_channels, output_size) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, cnn_conv_mid_channels, kernel_size=3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(cnn_conv_mid_channels, cnn_conv_mid_channels, kernel_size=3, padding=1),
            nn.ReLU(), 
            nn.Conv2d(cnn_conv_mid_channels, cnn_conv_mid_channels, kernel_size=3, padding=1),
            nn.ReLU(), 
            nn.Conv2d(cnn_conv_mid_channels, cnn_out_channels, kernel_size=3, padding=1)
        )
        adaptive_output_size = (1, 1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(adaptive_output_size)

        input_size = cnn_out_channels * adaptive_output_size[0] * adaptive_output_size[1]
        self.fc = nn.Sequential(
            nn.Linear(input_size, cnn_fc_hidden_layer_size), 
            nn.ReLU(), 
            nn.Linear(cnn_fc_hidden_layer_size, cnn_fc_hidden_layer_size), 
            nn.ReLU(), 
            nn.Linear(cnn_fc_hidden_layer_size, output_size)
        )
        self._initialize_weights()

    def forward(self, x: torch.Tensor):
        x: torch.Tensor = self.conv(x)
        x: torch.Tensor = self.adaptive_pool(x)
        x: torch.Tensor = x.flatten(1)
        x: torch.Tensor = self.fc(x)
        return x
    
    def save(self, filename='cnn_model.pth'):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        weights_dir = os.path.join(script_dir, 'weights')
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)
        
        full_path = os.path.join(weights_dir, filename)
        
        torch.save(self.state_dict(), full_path)
        print(f"Model saved to {full_path}")

    def load(self, filename='cnn_model.pth', device=None):
        script_dir = os.path.dirname(os.path.abspath(__file__))

        full_path = os.path.join(script_dir, 'weights', filename)

        if not os.path.exists(full_path):
            print(f"Model file not found at {full_path}")
            return False

        try:
            self.load_state_dict(torch.load(full_path, weights_only=True, map_location=device))
            self.eval()
            print(f"Model loaded from {full_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)
