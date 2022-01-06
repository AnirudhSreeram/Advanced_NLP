import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as bp

class Net(nn.Module):
    def __init__(self, inp, hidden, n_classes, hact, act):
        super(Net, self).__init__()
        self.hidden = hidden                                # init hidden layers
        self.classes = n_classes                            # init number of classes
        self.fc1 = nn.Conv2d(1, 32, 3, stride=2)            # init input layer
        self.fc2 = nn.Conv2d(32, 32 , kernel_size=(4,3 ))           # init hiddenlayer 1
        #self.pool = nn.MaxPool2d(3, stride=2)
        self.fc3 = nn.Conv2d(32, 1 , kernel_size=(1,12), stride=2)     # init output layer
        self.s = nn.Sigmoid()                               # init activation
        self.relu = nn.ReLU()                               # init activation
        self.tanh = nn.Tanh()                               # init activation
        self.softMax = nn.Softmax(dim=1)                    # init activation
        # check for activation function and select the corresponding activation for hidden layers
        if hact == "tanh":
            self.hact = self.tanh
        else:
            self.hact = self.relu
        # Check and select the output activation
        if act == "sigmoid":
            self.act = self.s
        else:
            self.act = self.softMax
        
    ############# perform forward pass #################
    def forward(self, x):
        x = torch.unsqueeze(x,1)
        x = self.hact(self.fc1(x))  # pass throught the first layer
        x = self.hact(self.fc2(x))  # pass throught the first layer
        x = self.fc3(x)  # pass through second layer
        x = self.act(torch.squeeze(x,2).squeeze(1))
        return x

    