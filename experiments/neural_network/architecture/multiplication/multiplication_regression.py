from torch import nn
from experiments.neural_network.architecture import layer


class MultiplicationRegression(nn.Module):
    def __init__(self, dim: int, depth: int, width: int):
        super(MultiplicationRegression, self).__init__()
        self.mlp = layer.MultilayerPerceptron([dim, *[width for _ in range(depth)]], non_linear=nn.ReLU,
                                              norm=nn.BatchNorm1d)
        self.output = nn.Linear(width, 1)

    def forward(self, x):
        return self.output(self.mlp(x))
