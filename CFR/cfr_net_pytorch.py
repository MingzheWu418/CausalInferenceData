import torch
import torch.nn as nn
import numpy as np


class RescalingLayer(nn.Module):
    """
    A custom layer for rescaling
    """

    def __init__(self, size_in, size_out):
        super(RescalingLayer, self).__init__()
        self.size_in, self.size_out = size_in, size_out
        weights = 1.0 / size_in * torch.ones([size_in])
        self.weights = nn.Parameter(weights)

    def forward(self, x):
        return torch.mm(x, self.weights.t())


class MyLinearLayer(nn.Module):
    """
    A custom layer with specified weights and bias
    """

    def __init__(self, size_in, size_out, weight_init):
        super(MyLinearLayer, self).__init__()
        self.size_in, self.size_out = size_in, size_out
        weights = torch.empty(size_out, size_in)
        self.weight = nn.Parameter(weights)
        bias = torch.empty(1, size_out)
        self.bias = nn.Parameter(bias)
        nn.init.normal_(self.weight, std=weight_init / np.sqrt(size_in))
        nn.init.normal_(self.bias, std=weight_init / np.sqrt(size_in))

    def forward(self, x):
        # print(x)
        w_mul_x = torch.mm(x, torch.Tensor(self.weight.t()))
        return torch.add(w_mul_x, self.bias)


class CFR(nn.Module):
    def __init__(self, dims, dropout_in, dropout_out, FLAGS):
        super(CFR, self).__init__()
        dim_input = dims[0]
        dim_in = dims[1]
        dim_out = dims[2]

        self.weights_pred = []
        self.weights_in = []
        self.weights_out = []

        n_in = FLAGS['n_in']
        n_out = FLAGS['n_out']
        weight_init = FLAGS['weight_init']
        if n_in == 0 or (n_in == 1 and FLAGS['varsel']):
            dim_in = dim_input
        if n_out == 0:
            dim_out = dim_in+1

        ''' Construct input/representation layers '''
        if FLAGS['varsel']:
            ''' If using variable selection, first layer is just rescaling'''
            self.in0 = RescalingLayer(dim_input, dim_in)
        else:
            self.in0 = MyLinearLayer(dim_input, dim_in, weight_init)
            # self.linear0 = nn.Linear(dim_input, dim_in)
            self.layers_in = nn.ModuleList()
        for i in range(n_in - 1):
            self.layers_in.append(MyLinearLayer(dim_in, dim_in, weight_init))
            # self.layers.append(nn.Linear(dim_in, dim_in))
            self.layers_in.append(nn.ReLU())
            self.layers_in.append(nn.Dropout(dropout_in))

        ''' Construct output/regression layers '''
        self.out0 = MyLinearLayer(dim_in+1, dim_out, weight_init)
        self.layers_out = nn.ModuleList()
        for i in range(n_out - 1):
            self.layers_out.append(MyLinearLayer(dim_out, dim_out, weight_init))
            self.layers_out.append(nn.ReLU())
            self.layers_in.append(nn.Dropout(dropout_out))

        ''' Output Layer '''
        self.output = MyLinearLayer(dim_out, 1, weight_init)

    def forward(self, x, t):
        x = torch.Tensor(x)
        t = torch.Tensor(t)
        x = self.in0(x)
        for layer in self.layers_in:
            x = layer(x)
        rep = x.clone()
        # print(x.shape)
        # print(t.shape)
        x = torch.cat([x, t], dim=1)
        x = self.out0(x)
        for layer in self.layers_out:
            x = layer(x)
        return self.output(x), rep

    def get_weights(self):
        self.weights_pred = self.output.weight
        self.weights_in.append(self.in0.weight)
        for layer in self.layers_in:
            try:
                self.weights_in.append(layer.weight)
            except AttributeError:
                pass
                # print(str(layer) + " has no attribute 'weight'")
        self.weights_out.append(self.out0.weight)
        for layer in self.layers_out:
            try:
                self.weights_out.append(layer.weight)
            except AttributeError:
                pass
                # print(str(layer) + " has no attribute 'weight'")
        return self.weights_pred, self.weights_in, self.weights_out
