"""
Docstring to be added later.
This module constains nn block classes.
"""

class Prob_Block(nn.Module):
    """
    TODO: UPDATE DOCSTRING AT LATER DATE
    Probabilistic block for neural network. Assumes input is a tensor of shape
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        hidden_layers: list[int],
        act_fn: nn.functional = nn.Tanh(),
        num_realizations: int = 1,
    ):
        super(Prob_Block, self).__init__()
        # Store params
        self.d_in = d_in
        self.d_out = d_out
        self.hidden_layers = hidden_layers
        self.act_fn = act_fn
        self.num_realizations = num_realizations
        # Build architecture
        Block = []
        neurons_in = d_in
        for neurons in hidden_layers:
            Block.append(nn.Linear(neurons_in, neurons))
            Block.append(act_fn)
            neurons_in = neurons
        Block.append(nn.Linear(neurons_in, 2 * d_out))  # Probabilistic output
        architecture = nn.Sequential(*Block)
        for layer in architecture:  # Initialize via Xavier uniform
            if isinstance(layer, nn.Linear):
                torch.nn.initxavier_uniform_(layer.weight)
                torch.nn.initzeros_(layer.bias)  # Initialize biases to zero
        self.architecture = architecture

    def forward(self, x):
        """
        Runs through network and returns distribution.
        """
        temp = self.architecture(x)
        mu, log_var = torch.chunk(temp, 2, dim=-1)
        sigma = torch.exp(0.5 * log_var)
        return torch.distributions.normal.Normal(mu, sigma)

    def predict(self, x):
        """
        Provides mean prediction.
        """
        temp = self.architecture(x)
        mu, log_var = torch.chunk(temp, 2, dim=-1)
        return mu

    def sample(self, x):
        """
        Samples once via reparameterization trick.
        """
        temp = self.architecture(x)
        mu, log_var = torch.chunk(temp, 2, dim=-1)
        return reparameterization_trick(mu, log_var)
    

class Det_Block(nn.Module):
    """
    TODO: Remove realizations (never helped anyways)
    Deterministic model block
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        hidden_layers: list[int],
        act_fn: nn.functional = nn.Tanh(),
        num_realizations: int = 1,
    ):
        super(Det_Block, self).__init__()
        # Store params
        self.d_in = d_in
        self.d_out = d_out
        self.hidden_layers = hidden_layers
        self.act_fn = act_fn
        self.num_realizations = num_realizations
        # Build architecture
        Block = []
        neurons_in = d_in
        for neurons in hidden_layers:
            Block.append(nn.Linear(neurons_in, neurons))
            Block.append(act_fn)
            neurons_in = neurons
        Block.append(nn.Linear(neurons_in, d_out))  # Deterministic output
        architecture = nn.Sequential(*Block)
        for layer in architecture:  # Initialize via Xavier uniform
            if isinstance(layer, nn.Linear):
                torch.nn.initxavier_uniform_(layer.weight)
                torch.nn.initzeros_(layer.bias)  # Initialize biases to zero
        self.architecture = architecture

    def forward(self, x):
        """
        Runs through network and returns distribution.
        """
        temp = self.architecture(x)
        return temp

    def predict(self, x):
        """
        Provides mean prediction.
        """
        return self(x)

    def sample(self, x):
        """
        Samples once via reparameterization trick.
        """
        return self(x)