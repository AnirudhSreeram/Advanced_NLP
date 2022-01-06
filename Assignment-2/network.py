import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as bp

class Net(nn.Module):
    def __init__(self, inp, hidden, n_classes, hact, act):
        super(Net, self).__init__()
        self.hidden = hidden                                # init hidden layers
        self.classes = n_classes                            # init number of classes
        self.fc1 = nn.Linear(inp, self.hidden)              # init input layer
        self.fc2 = nn.Linear(self.hidden, self.hidden)      # init hiddenlayer 1
        self.fc3 = nn.Linear(self.hidden, self.classes)     # init output layer
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
        x = self.hact(self.fc1(x))  # pass throught the first layer
        x = self.hact(self.fc2(x))  # pass through second layer
        x = self.act(self.fc3(x))   # output layer
        return x