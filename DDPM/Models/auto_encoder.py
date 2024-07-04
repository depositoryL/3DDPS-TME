import torch
from torch import nn


class AE(nn.Module):
    def __init__(self, in_dim=529, hidden_size_sqrt=12, dp_ratio=0.):
        super(AE, self).__init__()
        self.hidden_size_sqrt = hidden_size_sqrt
        self.hidden_size = hidden_size_sqrt ** 2

        self.embedding = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dp_ratio),
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            nn.Tanh()
        )

        self.recovery = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dp_ratio),
            nn.Linear(in_features=self.hidden_size, out_features=in_dim),
            nn.Sigmoid()
        )

    @torch.no_grad()
    def update(self, scale, shift):
        self.scale, self.shift = scale, shift

    def recover(self, input, unnormalize=False):
        device = input.device
        input = input.reshape(-1, self.hidden_size)
        output = self.recovery(input)
        if unnormalize:
            output = (output - self.shift.to(device)) / self.scale.to(device)
        return output
    
    def encode(self, input, normalize=False):
        device = input.device
        if normalize:
            input = input * self.scale.to(device) + self.shift.to(device)
        output = self.embedding(input)
        output = output.reshape(-1, 1, self.hidden_size_sqrt, self.hidden_size_sqrt)
        return output

    def forward(self, input):
        output = self.embedding(input)
        return self.recovery(output)
