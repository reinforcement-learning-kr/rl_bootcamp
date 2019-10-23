import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

def identity(x):
    """Return input without any change."""
    return x


class MLP(nn.Module):
    def __init__(self, 
                 input_size, 
                 output_size, 
                 hidden_sizes=(128,128), 
                 activation=F.relu,
                 output_activation=identity,
                 use_output_layer=True,
    ):
        super(MLP, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.output_activation = output_activation
        self.use_output_layer = use_output_layer

        # Set hidden layers
        self.hidden_layers = nn.ModuleList()
        in_size = self.input_size
        for next_size in self.hidden_sizes:
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            self.hidden_layers.append(fc)

        # Set output layers
        if self.use_output_layer:
            self.output_layer = nn.Linear(in_size, self.output_size)
        else:
            self.output_layer = identity

    def forward(self, x):
        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x))
        x = self.output_activation(self.output_layer(x))
        return x


class FlattenMLP(MLP):
    def forward(self, x, a):
        q = torch.cat([x,a], dim=-1)
        return super(FlattenMLP, self).forward(q)


LOG_STD_MAX = 2
LOG_STD_MIN = -20

class GaussianPolicy(MLP):
    def __init__(self, 
                 input_size, 
                 output_size, 
                 hidden_sizes=(128,128),
                 activation=F.relu,
    ):
        super(GaussianPolicy, self).__init__(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
            use_output_layer=False,
        )

        in_size = hidden_sizes[-1]

        # Set output layers
        self.mu_layer = nn.Linear(in_size, output_size)
        self.log_std_layer = nn.Linear(in_size, output_size)

    def clip_but_pass_gradient(self, x, l=-1., u=1.):
        clip_up = (x > u).float()
        clip_low = (x < l).float()
        clip_value = (u - x)*clip_up + (l - x)*clip_low
        return x + clip_value.detach()

    def apply_squashing_func(self, mu, pi, log_pi):
        mu = torch.tanh(mu)
        pi = torch.tanh(pi)
        # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
        log_pi -= torch.sum(torch.log(self.clip_but_pass_gradient(1 - pi.pow(2), l=0., u=1.) + 1e-6), dim=-1)
        return mu, pi, log_pi

    def forward(self, x):
        x = super(GaussianPolicy, self).forward(x)
        
        mu = self.mu_layer(x)
        log_std = torch.tanh(self.log_std_layer(x))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        std = torch.exp(log_std)
        
        dist = Normal(mu, std)
        pi = dist.rsample() # reparameterization trick (mean + std * N(0,1))
        log_pi = dist.log_prob(pi).sum(dim=-1)
        mu, pi, log_pi = self.apply_squashing_func(mu, pi, log_pi)
    
        return mu, pi, log_pi