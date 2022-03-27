from io import open
import glob
import os
import torch
import torch.nn as nn
import math

class LRNLayer(nn.Module):
    
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor
    bias: torch.Tensor

    def __init__(self, in_features: int, out_features: int, rec = True, bias = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LRNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if (rec):
            self.rec_weight = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.rec_weight = None
        if (bias):
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.rec_weight is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.rec_weight, -bound, bound)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor, prev_hidden: torch.Tensor) -> torch.Tensor:
        if self.rec_weight is not None:
            return (nn.functional.linear(input, self.weight, self.bias) + torch.mul(prev_hidden, self.rec_weight))
        else:
            return (nn.functional.linear(input, self.weight, self.bias))

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, rec={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.rec_weight is not None
        )

class LRN(nn.Module):
                    
    def __init__(self, input_size, hidden_layers_sizes, output_size):
                    
        super(LRN, self).__init__()

        self.hidden_layers_sizes = hidden_layers_sizes

        self.layers = []
        
        # self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        # self.i2o = nn.Linear(input_size + hidden_size, output_size)

        self.layers.append(LRNLayer(input_size, hidden_layers_sizes[0]))
        for i in range(len(hidden_layers_sizes)-1):
            self.layers.append(LRNLayer(hidden_layers_sizes[i], hidden_layers_sizes[i+1]))
        
        self.layers.append(LRNLayer(hidden_layers_sizes[-1], output_size, rec=False))
        
        self.Activation = nn.Softmax(dim=1)

    def forward(self, input, previous_state): # we have the previous hidden layers

        new_state = []

        new_state.append(self.Activation(self.layers[0](input, previous_state[0])))

        for i in range(1, len(self.layers)-1):
            new_state.append(self.Activation(self.layers[i](new_state[i-1], previous_state[i])))

        output = self.Activation(self.layers[-1](new_state[-1], previous_state[-1]))
        
        return output, new_state

    def initHidden(self):

        zero_state = []

        for layer_size in self.hidden_layers_sizes:
            zero_state.append(torch.zeros(1, layer_size))

        return zero_state


lrn = LRN(1, [10, 20], 1)

hidden = lrn.initHidden()

print(hidden)

input = torch.zeros(1, 1)

output, next_hidden = lrn(input, hidden)

print(output)
print(next_hidden)

costFunction = nn.MSELoss()

# def train (current, voltages):

#     hidden = lrn.initHidden()

#     lrn.zero_grad()

#     for i in range(voltages):
#         output, hidden = run(
