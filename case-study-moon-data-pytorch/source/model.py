import torch
import torch.nn as nn
import torch.nn.functional as F

## TODO: Complete this classifier
class SimpleNet(nn.Module):
    
    ## TODO: Define the init function
    def __init__(self, input_dim, hidden_dim, output_dim):
        '''
        Defines layers of a neural network.
           :param input_dim: Number of input features
           :param hidden_dim: Size of hidden layer(s)
           :param output_dim: Number of outputs
        '''
        super(SimpleNet, self).__init__()
        
        # define all layers, here
        # self.input_dim, self.hidden_dim, self.output_dim = input_dim, hidden_dim, output_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.drop = nn.Dropout(0.20)
        self.sig = nn.Sigmoid()
    
    ## TODO: Define the feedforward behavior of the network
    def forward(self, x):
        '''
        Feedforward behavior of the net.
           :param x: A batch of input features
           :return: A single, sigmoid activated value
        '''
        # your code, here
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.sig(x)
        return x