import torch
import torch.nn as nn

class Expression(nn.Module):
    def __init__(self, func):
        super(Expression, self).__init__()
        self.func = func
    def forward(self, input):
        return self.func(input)

class simpleMLP(nn.Module):
    def __init__(self, i_c=1, n_c=10):
        super(simpleMLP, self).__init__()

        self.flatten = Expression(lambda tensor: tensor.view(tensor.shape[0], -1))
        self.fc1 = nn.Linear(28*28, 256, bias=True)
        self.fc2 = nn.Linear(256, 128, bias=True)
        self.fc3 = nn.Linear(128, n_c, bias=True)

    def forward(self, x_i, _eval=False):
        x_o = self.flatten(x_i)
        x_o = torch.relu(self.fc1(x_o))
        x_o = torch.relu(self.fc2(x_o))

        return self.fc3(x_o)
