# import torch
import torch.nn as nn

class Helper(nn.Module):
    """
    Change Pm feature to Laten Vector with d = 64
    Args:
        nn (_type_): _description_
    """
    
    def __init__(self, input_dim=3, d_model=64):
        super().__init__()
        
        # Using MLP(Multi-layer Perceptron) for data
        self.projector = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, d_model),
            nn.LayerNorm(d_model) # On dinh du lieu
        )
        
    def forward(self, x):
        # x shape: (batch_size, num_pms, 3)
        return self.projector(x)