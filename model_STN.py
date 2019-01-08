# Author- Aishwarya Budhkar
# STN model

import torch
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self,num_classes=43):
        super(Net, self).__init__()
        self.in_planes = 64

        # STN module for color variance
        self.conv1 = nn.Conv2d(3, 10, kernel_size=1, stride=1, padding=2)
        self.conv1_drop = nn.Dropout2d(0.05)
        self.ln_drop=nn.Dropout2d(0.1)
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 3, kernel_size=1, stride=1, padding=2)
        self.conv2_drop = nn.Dropout2d(0.05)
        self.bn2 = nn.BatchNorm2d(3)

        # STN module for spatial variance
        self.conv3 = nn.Conv2d(3, 16, kernel_size=7, stride=1, padding=1)
        self.conv4 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=1)
        self.conv5 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.l1 = nn.Linear(2304, 128)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64,6 )
        self.conv6 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2)
        self.bn6 = nn.BatchNorm2d(16)
        self.conv7 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.bn7 = nn.BatchNorm2d(32)
        self.conv8 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn8 = nn.BatchNorm2d(64)
        self.conv9 = nn.Conv2d(64, 96, kernel_size=5, stride=1, padding=2)
        self.bn9 = nn.BatchNorm2d(96)
        self.conv10 = nn.Conv2d(96, 128, kernel_size=5, stride=1, padding=2)
        self.bn10 = nn.BatchNorm2d(128)
        self.conv11 = nn.Conv2d(128, 192, kernel_size=5, stride=1, padding=2)
        self.bn11 = nn.BatchNorm2d(192)
        self.conv12 = nn.Conv2d(192, 256, kernel_size=5, stride=1, padding=2)
        self.bn12 = nn.BatchNorm2d(256)
        self.conv13 = nn.Conv2d(256, 128, kernel_size=5, stride=1, padding=2)
        self.bn13 = nn.BatchNorm2d(128)
        self.conv14 = nn.Conv2d(128, 64,kernel_size=5, stride=1, padding=2)
        self.bn14 = nn.BatchNorm2d(64)
        self.fc3 = nn.Linear(64,num_classes)
        self.elu=nn.ELU()

    def _make_layer(self, block, planes, num_blocks, stride):

        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.bn1(F.leaky_relu(self.conv1_drop(self.conv1(x))))
        out = self.bn2(F.leaky_relu(self.conv2_drop(self.conv2(out))))
        out = F.max_pool2d(self.conv3(out), 2)
        out = F.max_pool2d(self.conv4(out), 2)
        out = F.max_pool2d(self.conv5(out), 2)
        out = out.view(out.size(0),-1)
        out = self.elu(self.l1(out))
        out = self.elu(self.l2(out))
        out = self.l3(out)
        out = out.view(-1, 2, 3)
        grid = F.affine_grid(out, x.size())
        x = F.grid_sample(x, grid)

        out = self.bn6(F.relu(self.conv2_drop(self.conv6(x))))
        out = self.bn7(F.relu(self.conv2_drop(self.conv7(out))))
        out = self.bn8(F.relu(self.conv2_drop(self.conv8(out))))
        out = self.bn9(F.relu(self.conv2_drop(self.conv9(out))))
        out = self.bn10(F.relu(self.conv2_drop(self.conv10(out))))
        out = F.max_pool2d(self.bn11(F.relu(self.conv2_drop(self.conv11(out)))),2)
        out = F.max_pool2d(self.bn12(F.relu(self.conv2_drop(self.conv12(out)))),2)
        out = self.bn13(F.relu(self.conv2_drop(self.conv13(out))))
        out = F.max_pool2d(self.bn14(F.relu(self.conv2_drop(self.conv14(out)))),8)
        out = out.view(out.size(0),-1)
        out = self.fc3(out)
        out = self.ln_drop(out)
        return F.log_softmax(out)