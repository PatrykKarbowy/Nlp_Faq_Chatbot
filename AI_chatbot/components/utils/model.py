import torch
import torch.nn as nn

class ChatNet(nn.Module):
    """Custom Neural Network model used for training FAQ Chatbot.

    Args:
        input_size (int): First layer input features.
        hidden_size (int): Hidden layer input size.
        output_size (int): Model output size.
    """
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        output_size: int
        ) -> None:
        super(ChatNet, self).__init__()
        self.linear_1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.linear_3 = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Neural Network Model forward call.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        out = self.linear_1(x)
        out = self.relu(out)
        out = self.linear_2(out)
        out = self.relu(out)
        out = self.linear_3(out)
        
        return out   