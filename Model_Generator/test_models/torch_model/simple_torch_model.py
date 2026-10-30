import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)

def get_model():
    return SimpleNet()

def main():
    model = get_model()
    x = torch.randn(1, 4)
    print(model(x))