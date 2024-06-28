import torch
import torch.nn as nn


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # Input is (60,1)
        self.shared = nn.Sequential(
            nn.Conv1d(2,4,3),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(4,8,3),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(8,16,3),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(16,32,3),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.fc = nn.Sequential(
            nn.Linear(1408, 256),
            nn.ReLU(),
            # nn.BatchNorm1d(512),
            # nn.Linear(512, 256),
            # nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.5),
            nn.Linear(32, 1)
        
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x1 = x1.permute(0,2,1)
        x2 = x2.permute(0,2,1)
        x1 = self.shared(x1)
        x1 = x1.view(x1.size(0), -1)
        x2 = self.shared(x2)
        x2 = x2.view(x2.size(0), -1)
        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
    
class SiameseNetworkToyConv(nn.Module):
    def __init__(self):
        super(SiameseNetworkToyConv, self).__init__()
        # Input is (60,1)
        self.shared = nn.Sequential(
            nn.Conv1d(2,4,3),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(4,8,3),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(8,16,3),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.fc = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.5),
            nn.Linear(32, 1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x1 = x1.permute(0,2,1)
        x2 = x2.permute(0,2,1)
        x1 = self.shared(x1)
        x1 = x1.view(x1.size(0), -1)
        x2 = self.shared(x2)
        x2 = x2.view(x2.size(0), -1)
        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
    
class SiameseNetworkToy(nn.Module):
    def __init__(self):
        super(SiameseNetworkToy, self).__init__()
        # Input is (60,1)
        self.shared = nn.Sequential(
            nn.Linear(60, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.5),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Dropout(0.5),
        )
        self.fc = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.5),
            nn.Linear(32, 1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x1 = x1.flatten(start_dim=1, end_dim=-1)
        x2 = x2.flatten(start_dim=1, end_dim=-1)
        x1 = self.shared(x1)
        x2 = self.shared(x2)

        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
    
