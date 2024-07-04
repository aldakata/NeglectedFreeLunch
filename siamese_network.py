import torch
import torch.nn as nn

class SiameseNetworkConv(nn.Module):
    def __init__(self):
        super(SiameseNetworkConv, self).__init__()
        # Input is 
        # (60,1) of mouse record
        # (2,1) of estimate time, and worker id
        self.shared_conv = nn.Sequential(
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
        
        self.shared_fc = nn.Sequential(
            nn.Linear(706, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),     
            nn.BatchNorm1d(256),
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.5),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x0, x1, t0, t1, w0, w1):
        x0 = self.shared_conv(x0)
        x0 = x0.view(x0.size(0), -1)
        x0 = self.shared_fc(torch.cat((x0, t0, w0), dim=1))

        x1 = self.shared_conv(x1)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.shared_fc(torch.cat((x1, t1, w1), dim=1))


        x = torch.cat((x0, x1), dim=1)
        x = self.fc(x)
        return x