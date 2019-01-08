# Author- Aishwarya Budhkar
# 3 layer model

import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43 # GTSRB as 43 classes

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv2_drop = nn.Dropout2d()
        self.conv1 = nn.Conv2d(3, 100, kernel_size=7)
        self.conv2 = nn.Conv2d(100, 150, kernel_size=4)
        self.conv3 = nn.Conv2d(150, 250, kernel_size=4)
        self.conv2_drop = nn.Dropout2d()
        self.conv1_bn = nn.BatchNorm2d(100)
        self.conv2_bn = nn.BatchNorm2d(150)
        self.conv3_bn = nn.BatchNorm2d(250)
        self.fc1 = nn.Linear(2250, 300)
        self.fc2 = nn.Linear(300, nclasses)


    def forward(self, x):_drop(self.conv2(x)), 2))            
            x = F.max_pool2d(F.relu(self.conv1_bn(self.conv1(x))), 2)
            x = F.max_pool2d(F.relu(self.conv2_bn(self.conv2(x))), 2)
            x = F.max_pool2d(F.relu(self.conv3_bn(self.conv3(x))), 2)
            x = x.view(x.size(0), -1)

            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x)

