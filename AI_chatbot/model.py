import torch
import torch.nn as nn

class ChatNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatNet, self).__init__()
        self.linear_1 = nn.Linear(in_features = input_size, out_features = hidden_size)
        self.linear_2 = nn.Linear(in_features = hidden_size, out_features = hidden_size)
        self.linear_3 = nn.Linear(in_features = hidden_size, out_features = output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.linear_1(x)
        out = self.relu(out)
        out = self.linear_2(out)
        out = self.relu(out)
        out = self.linear_3(out)
        
        return out   