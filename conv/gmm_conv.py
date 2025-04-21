import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from .inits import reset, glorot, zeros

EPS = 1e-15


'This function constructs a Gaussian Mixture Model with a Convolutional Layer. The messages between nodes are weighted using Gaussian kernels on pseudo-coordinates.'
'The learnable parameters here are the mean and standard deviation.'

class GMMConv(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dim,
                 kernel_size,
                 bias=True,
                 **kwargs):
        super(GMMConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.kernel_size = kernel_size

        self.lin = torch.nn.Linear(in_channels,
                                   out_channels * kernel_size,
                                   bias=False)
        self.mu = Parameter(torch.Tensor(kernel_size, dim))
        self.sigma = Parameter(torch.Tensor(kernel_size, dim))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()


    'Weights are initialized in this function'
    def reset_parameters(self):
        glorot(self.mu)
        glorot(self.sigma)
        zeros(self.bias)
        reset(self.lin)


   
    'Perform a linear transformation on the nodes'
    def forward(self, x, edge_index, pseudo):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        pseudo = pseudo.unsqueeze(-1) if pseudo.dim() == 1 else pseudo

        x = self.lin(x) 
        return self.propagate(edge_index, x=x, pseudo=pseudo)

    'Reshapes the message and calculates the gaussian weights based on the pseudo-coordinates'
    def message(self, x_j, pseudo):
        (E, D), K = pseudo.size(), self.mu.size(0)

        x_j = x_j.view(-1, K, self.out_channels)  

        gaussian = -0.5 * (pseudo.view(E, 1, D) - self.mu.view(1, K, D))**2
        gaussian = gaussian / (EPS + self.sigma.view(1, K, D)**2)
        gaussian = torch.exp(gaussian.sum(dim=-1, keepdim=True))  # [E, K, 1]

        return (x_j * gaussian).sum(dim=1)  # [E, out_channels]


    def __repr__(self):
        return '{}({}, {}, kernel_size={})'.format(self.__class__.__name__,
                                                   self.in_channels,
                                                   self.out_channels,
                                                   self.kernel_size)
