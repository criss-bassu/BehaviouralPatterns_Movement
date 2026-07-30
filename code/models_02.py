import torch
import torch.nn as nn

class MLPEncoder(nn.Module):
    def __init__(self, d, hours, hidden_dim = 256, rep_dim = 128, dropout = 0.3):
        super().__init__()
        # size of the array (a matrix will be flatten into a vector)
        input_dim = hours * d # 168 x num_descriptores
        # MLP definition
        # Layers are applied sequentially, with ReLU activations and dropout in between
        self.net = nn.Sequential(
            nn.Flatten(), # Flatten the input tensor to a vector
            nn.Linear(input_dim, hidden_dim), # First linear layer (input_dim -> hidden_dim)
            nn.ReLU(), # Activation function
            nn.Dropout(dropout), # 30% of activations are dropped (reduces overfitting)
            nn.Linear(hidden_dim, hidden_dim), # Second linear layer
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, rep_dim), # Final linear layer. Produces the representation. (hidden_dim -> rep_dim)
            nn.ReLU()
        )

    # x = (batch_size, hours, d)
    def forward(self, x):
        return self.net(x) # Runs the MLP network. Returns (batch_size, rep_dim)


class CNNEncoder(nn.Module):
    def __init__(self, d, channels = 64, rep_dim = 128, dropout = 0.3, kernel_size = 5):
        super().__init__()
        padding = kernel_size // 2
        # 1D-CNN definition
        self.conv = nn.Sequential(
            # in_channels = number of input features = d descriptors
            # kernel_size = looks at 5 consecutive hours
            # out_channels = number of patterns the CNN can learn
            nn.Conv1d(in_channels = d, out_channels = channels, kernel_size = kernel_size, padding = padding),
            nn.ReLU(),
            nn.BatchNorm1d(channels), # Normalization layer to stabilize training
            nn.Dropout(dropout),
            # Previous convolutional features as input.
            nn.Conv1d(in_channels = channels, out_channels = channels, kernel_size = kernel_size, padding = padding),
            nn.ReLU(),
            nn.BatchNorm1d(channels),
            # Converts a time sequence into one summary vector per sample (global average pooling)
            nn.AdaptiveAvgPool1d(1)
        )
        # Converts the convolutional summary into the final representation size: (batch_size, channels, 1) -> (batch_size, channels)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, rep_dim), # Produces the representation: (batch_size, channels) -> (batch_size, rep_dim)
            nn.ReLU()
        )

    # x = (batch_size, hours, d)
    def forward(self, x):
        x = x.transpose(1, 2) # Expected input: (batch_size, hours, d) -> (batch_size, d, hours)
        h = self.conv(x) # Passes the transposed input through the convolutional block
        return self.proj(h) # returns (batch_size, rep_dim)


class GRUEncoder(nn.Module):
    def __init__(self, d, hidden_dim = 128, rep_dim = 128, num_layers = 1, dropout = 0.0):
        super().__init__()
        # GRU definition
        self.gru = nn.GRU(
            input_size = d, # descriptors
            hidden_size = hidden_dim,
            num_layers = num_layers,
            batch_first = True, # The input tensor has a initial batch size: (batch_size, hours, d)
            dropout = dropout if num_layers > 1 else 0.0 # If there is only one layer, dropout inside the GRU won't be meaningful
        )
        # Converts the GRU output into the final representation size
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, rep_dim), # (batch_size, hidden_dim) -> (batch_size, rep_dim)
            nn.ReLU()
        )

    # x = (batch_size, hours, d)
    def forward(self, x):
        # Runs the GRU
        output, h_n = self.gru(x) # output = (batch_size, hours, hidden_dim); h_n = (num_layers, batch_size, hidden_dim)
        final_hidden = h_n[-1] # Takes the final hidden state from the last GRU layer
        return self.proj(final_hidden) # (batch_size, rep_dim)

# Takes the encoded weekly representation plus the clinical covariates and produces one prediction per task
class MultiTaskHead(nn.Module):
    def __init__(self, tasks, rep_dim = 128, cov_dim = 3, hidden_dim = 128, dropout = 0.3):
        super().__init__()
        input_dim = rep_dim + cov_dim # input dimension: (batch_size, rep_dim + cov_dim)

        # Shared block by all tasks:
        # All tasks first use the same transformation before branching into separate output heads
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), # (batch_size, rep_dim + cov_dim) -> (batch_size, hidden_dim)
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # A linear head for each task, dynamically built from the list of tasks (binary + numeric)
        # Order is preserved
        self.heads = nn.ModuleDict({task: nn.Linear(hidden_dim, 1) for task in tasks}) # Dictionary of neural network layers

    # Takes the encoded weekly representation and the covariates, and produces one prediction per task
    def forward(self, h_week, cov):
        # Concatenates the weekly representation and covariates along the feature dimension
        # (batch_size, rep_dim) + (batch_size, cov_dim) -> (batch_size, rep_dim + cov_dim)
        z = torch.cat([h_week, cov], dim = 1)
        # Passes the concatenated input through the shared block
        # (batch_size, rep_dim + cov_dim) -> (batch_size, hidden_dim)
        z = self.shared(z)
        # Returns a dictionary with predictions for each task
        # Each prediction is squeezed to remove the singleton dimension: (batch_size, 1) -> (batch_size,)
        return {task: head(z).squeeze(1) for task, head in self.heads.items()} # Each task gets one prediction per sample

class WeeklyOutcomeModel(nn.Module):
    def __init__(self, encoder, tasks, rep_dim = 128, cov_dim = 3, head_hidden = 128, head_dropout = 0.3):
        super().__init__()
        self.encoder = encoder
        self.head = MultiTaskHead(
            tasks,
            rep_dim = rep_dim, 
            cov_dim = cov_dim, 
            hidden_dim = head_hidden, 
            dropout = head_dropout
        )

    # x = (batch_size, hours, d); cov = (batch_size, cov_dim)
    def forward(self, x, cov):
        h_week = self.encoder(x) # Passes the weekly sequence through the encoder
        return self.head(h_week, cov) # Passes the weekly representation and covariates into the multi-task head
