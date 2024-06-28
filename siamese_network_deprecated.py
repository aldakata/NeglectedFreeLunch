from torch import nn as nn
import torch

## Maybe try LSTM:
# https://github.com/yakhyo/pytorch-tutorials/blob/main/tutorials/03-intermediate/04-lstm-network/main.py
# Use convolution
class Siamese_network(nn.Module):
    # It should predict 0, 1.
    # 0 if 1st sample has higher loss
    # 1 if 2nd sample has higher loss
    def __init__(self):
        super().__init__()

        # input is 30x3 dim and output is 1 dim
        self.shared_features = nn.Sequential(
            nn.Linear(60, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
        )

        self.classification_head = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 1),
            nn.ReLU(),
            nn.BatchNorm1d(1),
        )

    def forward(self, x, y):
        x = x.flatten(start_dim=1, end_dim=-1)
        y = y.flatten(start_dim=1, end_dim=-1)
        x = self.shared_features(x)
        y = self.shared_features(y)

        return self.classification_head(x * y)



## Maybe try LSTM:
# https://github.com/yakhyo/pytorch-tutorials/blob/main/tutorials/03-intermediate/04-lstm-network/main.py
# Use convolution
class Siamese_network_toy(nn.Module):
    # It should predict 0, 1.
    # 0 if 1st sample has higher loss
    # 1 if 2nd sample has higher loss
    def __init__(self):
        super().__init__()

        # input is 30x3 dim and output is 1 dim
        self.shared_features = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.5),
        )

        self.classification_head = nn.Sequential(
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Dropout(0.5),
            nn.Linear(8, 1),
            nn.ReLU(),
            nn.BatchNorm1d(1),
        )

    def forward(self, x, y):
        x = x.flatten(start_dim=1, end_dim=-1)
        y = y.flatten(start_dim=1, end_dim=-1)
        x = self.shared_features(x)
        y = self.shared_features(y)

        return self.classification_head(x * y)
