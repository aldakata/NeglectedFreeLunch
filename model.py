import torch
import torch.nn as nn
# This models needs an extra head for the foreground point

class model_foregroud(nn.Module):
    def __init__(self, num_classes, resnet_head):
        self.resnet = resnet_head
        self.num_classes = num_classes
        self.foreground_head = 
        self.fc = torch.nn.Linear(1000, num_classes)
        self.fc2 = torch.nn.Linear(1000, 2)

    def forward(self, x):
        x = self.model(x)
        return self.fc(x), self.fc2(x)

    def parameters(self):
        return list(self.model.parameters()) + list(self.fc.parameters()) + list(self.fc2.parameters())