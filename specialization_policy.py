import torch
import torch.nn as nn
from vmas.simulator.environment import Environment

from scenarios.tasks_comms import ScenarioTaskComms


class SpecializationPolicy(nn.Module):

    def __init__(self,
                 obs_size,
                 actions_size,
                 output_shape,
                 hidden_size=32,
                 device='cpu',
                 ):
        
        super(SpecializationPolicy, self).__init__()
        
        # Define the network layers
        self.fc1 = nn.Linear(obs_size, hidden_size, device=device)  # Input to hidden layer
        self.relu = nn.ReLU()  # Activation function
        self.fc2 = nn.Linear(hidden_size, hidden_size, device=device)  # Hidden to output layer
        # TODO Consider multi-head policy here!
        self.out1 = nn.Linear(hidden_size, 2, device=device)  # Hidden to output layer
        self.out2 = nn.Linear(hidden_size, 2, device=device)  # Hidden to output layer
        
        self.softmax = nn.Softmax(dim=1)
        self.output_shape = output_shape
        
        self.fitness = None
    
    def forward(self, obs):
        
        # Pass data through the layers
        x = self.fc1(obs)  # Linear transformation
        x = self.relu(x)  # Apply ReLU activation
        x = self.fc2(x)  # Final linear transformation
        outs = [self.softmax(self.out1(x)), 
                self.softmax(self.out2(x))]
        # print("fc2 out", x)
        # x = torch.reshape(x, self.output_shape)
        # print("outs", outs)
        actions = outs
        
        # print("softmax:", actions)
        
        return actions
    