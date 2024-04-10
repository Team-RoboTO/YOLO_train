import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def findConv2dOutShape(hin, win, conv, pool=2):
    # get conv arguments
    kernel_size = conv.kernel_size
    stride = conv.stride
    padding = conv.padding
    dilation = conv.dilation

    hout = np.floor((hin+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1)
    wout = np.floor((win+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1]+1)

    if pool:
        hout /= pool
        wout /= pool

    return int(hout), int(wout)


class BinaryClassifier(nn.Module):

    def __init__(self, dimensions: tuple, num_classes, init_f=8, num_fc1=100, dropout_rate = 0.25) -> None:
        super().__init__()

        assert len(dimensions) == 3
        Cin, Hin, Win = dimensions
        self.dropout_rate = dropout_rate
        
        # Convolution Layers
        self.conv1 = nn.Conv2d(Cin, init_f, kernel_size=3)
        h, w = findConv2dOutShape(Hin, Win, self.conv1)

        self.conv2 = nn.Conv2d(init_f, 2*init_f, kernel_size=3)
        h, w = findConv2dOutShape(h, w, self.conv2)

        self.conv3 = nn.Conv2d(2*init_f, 4*init_f, kernel_size=3)
        h, w = findConv2dOutShape(h, w, self.conv3)

        self.conv4 = nn.Conv2d(4*init_f, 8*init_f, kernel_size=3)
        h, w = findConv2dOutShape(h, w, self.conv4)
        
        # compute the flatten size
        self.num_flatten = h * w * 8 * init_f
        self.fc1 = nn.Linear(self.num_flatten, num_fc1)
        self.fc2 = nn.Linear(num_fc1, num_classes)
    

    def forward(self, x):

        # Convolution & Pool Layers
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)

        x = x.view(-1, self.num_flatten)
        
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dropout_rate)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)