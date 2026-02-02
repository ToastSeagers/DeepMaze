import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        # Shared Feature Layer
        self.fc1 = nn.Linear(input_dim, 256)
        
        # Value Stream (V) - estimates how good the current state is
        self.fc_value = nn.Linear(256, 128)
        self.value = nn.Linear(128, 1)
        
        # Advantage Stream (A) - estimates benefit of each action
        self.fc_adv = nn.Linear(256, 128)
        self.advantage = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        
        # Value
        v = F.relu(self.fc_value(x))
        V = self.value(v)
        
        # Advantage
        a = F.relu(self.fc_adv(x))
        A = self.advantage(a)
        
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        Q = V + (A - A.mean(dim=1, keepdim=True))
        return Q
