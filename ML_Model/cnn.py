import iris
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from iris.cube import Cube

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        nlats=144
        nlons=192
        kernel_size=(3,3)
        stride=(2,2)
        padding=(0,0)
        output_padding=(1,1)
        input_size=(nlats,nlons)
        output_size=(int(np.floor((input_size[0] - kernel_size[0] + 2 * padding[0]) / stride[0]) + 1),int(np.floor((input_size[1] - kernel_size[1] + 2 * padding[1]) / stride[1]) + 1))
 
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=30,kernel_size=kernel_size,stride=stride,padding=padding)
        self.dropout1 = nn.Dropout2d(p=0.25)
        self.fc1 = nn.Linear(output_size[0]*output_size[1], 200)
        self.fc2 = nn.Linear(200, output_size[0]*output_size[1])
        self.unflatten = nn.Unflatten(2, output_size)
        self.convT1 = nn.ConvTranspose2d(in_channels=30,out_channels=1,kernel_size=kernel_size,stride=stride,output_padding=output_padding,padding=padding)
	    
        # Initialize the weights of the convolutional layers
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.convT1.weight)
    
    def forward(self, x):
        # Forward pass
        x = self.conv1(x)
        x = nn.LeakyReLU()(x)
        x = self.dropout1(x)
        x = nn.Flatten(start_dim = 2, end_dim = 3)(x)
        x = self.fc1(x)
        x = nn.Tanh()(x)
        x = self.fc2(x)
        x = nn.Tanh()(x)
        x = self.unflatten(x)
        x = nn.LeakyReLU()(x)
        x = self.convT1(x)
        x = nn.LeakyReLU()(x)
        return x

