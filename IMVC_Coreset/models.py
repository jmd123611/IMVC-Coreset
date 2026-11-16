import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, input_dim, *layer_dims):
        super(MLP, self).__init__()
        if len(layer_dims) < 2:
            raise ValueError('MLP requires hidden and output dimensions')
        dimensions = (input_dim, *layer_dims)
        layers = []
        for index, (in_features, out_features) in enumerate(zip(dimensions, dimensions[1:])):
            layers.append(nn.Linear(in_features, out_features))
            if index < len(dimensions) - 2:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
