from torch import nn


class MultilayerPerceptron(nn.Module):
    def __init__(self, channels_list, norm=None, non_linear=None, last_layer_act=True):
        super(MultilayerPerceptron, self).__init__()
        self.n_layer = len(channels_list) - 1
        if self.n_layer < 1:
            raise Exception("Wrong Number of Layer")
        print(f"Building MultiLayer Perceptron with :{self.n_layer}")
        layers_list = []
        for i in range(self.n_layer):
            layers_list.append(nn.Linear(channels_list[i], channels_list[i + 1], bias=norm is None))
            nn.init.xavier_normal_(layers_list[-1].weight)
            if norm is not None:
                layers_list.append(norm(channels_list[i + 1]))
            if non_linear is not None:
                if i == (self.n_layer - 1) and not last_layer_act:
                    pass
                else:
                    layers_list.append(non_linear())

        self.layer_sequential = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.layer_sequential(x)
