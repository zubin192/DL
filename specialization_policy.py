import torch
import torch.nn as nn
from vmas.simulator.environment import Environment

from scenarios.tasks_comms import ScenarioTaskComms

# ==== EA Functions ====

class SpecializationPolicy(nn.Module):

    def __init__(self,
                 obs_size,
                 num_agents, # num heads
                 num_modes, # num modalities
                 hidden_size=16,
                 device='cpu',
                 ):
        
        super(SpecializationPolicy, self).__init__()
        self.num_agents = num_agents
        self.num_modes = num_modes
        
        # Define the network layers
        self.fc1 = nn.Linear(obs_size, hidden_size, device=device)
        self.relu = nn.ReLU()  # Activation function
        self.fc2 = nn.Linear(hidden_size, hidden_size, device=device)
        # Multi-head policy here
        self.heads = [nn.Linear(hidden_size, num_modes, device=device) for _ in range(num_agents)]
        
        self.softmax0 = nn.Softmax(dim=0)
        self.softmax1 = nn.Softmax(dim=1)
        
        self.fitness = None
    
    def forward(self, obs):
        
        # Pass data through the layers
        x = self.fc1(obs)  # Linear transformation
        x = self.relu(x)  # Apply ReLU activation
        # x = self.fc2(x)  # Final linear transformation
        # x = self.relu(x)  # Apply ReLU activation
        heads_x = [head(x) for head in self.heads]
        # print("Head shape:", len(heads_x[0].shape))
        if len(heads_x[0].shape) > 1: # for batch dim > 1
            actions = [self.softmax1(i) for i in heads_x]
        else:
            actions = [self.softmax0(i) for i in heads_x]
        
        # print("Raw heads:\n", heads_x, "\nActions:\n", actions)
        
        return actions
    