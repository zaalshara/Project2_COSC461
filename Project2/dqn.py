import torch
import torch.nn as nn 
#nn Neural Network
#Linear and Relu are parts of nn to help AI learn


#Deep Q-Network for MountainCar environment
class DQN(nn.Module):

    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, state):
        return self.network(state)