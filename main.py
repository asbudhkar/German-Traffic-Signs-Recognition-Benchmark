# Author- Aishwarya Budhkar

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# Import for plotting graphs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Model Summary
from torchsummary import summary

import numpy as np

# Adjust Learning rate decay
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.1, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

torch.manual_seed(args.seed)

# Data Initialization and Loading
from data import initialize_data, data_transforms # data.py in the same folder
initialize_data(args.data) # extracts the zip files, makes a validation set

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/train_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=True, num_workers=1)
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/val_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=False, num_workers=1)

# Neural Network and Optimizer
# Define neural net in model.py so that it can be reused by the evaluate.py script
from model import Net

model = Net().cuda()
# Use Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001, eps=1e-2, amsgrad=True)
lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, verbose=True, threshold=0.001, threshold_mode='rel', cooldown=3, min_lr=0, eps=1e-02)

losses_train=[]
losses_val=[]
e_train=[]
acc_train=[]
def train(epoch):
    model.train().cuda()
    training_loss = 0
    correct=0
    loss_all=0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).cuda(), Variable(target).cuda()
        optimizer.zero_grad()
        output = model(data).cuda()
        loss = F.nll_loss(output, target).cuda()
        loss_all+=loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).sum()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
    training_loss = loss_all/len(train_loader)
    losses_train.append(training_loss)
    training_acc=100. * correct / len(train_loader)
    acc_train.append(training_acc)
    e_train.append(epoch)

e_val=[]
acc_val=[]
def validation():
    best_val_loss=np.inf
    model.eval().cuda()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda()
        output = model(data).cuda()
        validation_loss += F.nll_loss(output, target, size_average=False).cuda().data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    lr_scheduler.step(validation_loss)                   
    losses_val.append(validation_loss)
    is_best = validation_loss<best_val_loss
    best_val_loss = min(validation_loss, best_val_loss)
    if is_best:
       torch.save(model.state_dict(), 'm.pth')   
    e_val.append(epoch)
    validation_acc=100. * correct / len(val_loader.dataset)
    acc_val.append(validation_acc)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    validation()
    best_val_loss = np.inf
    model_file = 'model_' + str(epoch) + '.pth'
    torch.save(model.state_dict(), model_file)
    print('\nSaved model to ' + model_file + '. You can run `python evaluate.py --model' + model_file + '` to generate the Kaggle formatted csv file')
with open('train_losses.txt', 'w') as f:
    for item in losses_train:
        f.write("%s\n" % item)
with open('test_losses.txt', 'w') as f:
    for item in losses_val:
        f.write("%s\n" % item)
#Code for plotting
print(losses_val)
print(losses_train)
print(acc_val)
print(acc_train)
print(e_train)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(e_train,losses_train,'r-')

plt.xlabel('Epochs')
plt.ylabel('Losses')
fig.savefig('train_loss.png')
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(e_val,losses_val,'r-')
plt.xlabel('Epochs')

plt.ylabel('Losses')
fig.savefig('test_loss.png')
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(e_train,acc_train,'r-')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
fig.savefig('train_acc.png')
fig = plt.figure()
ax = fig.add_subplot(111)

plt.plot(e_val,acc_val,'r-')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
fig.savefig('test_acc.png')

summary(model,(3,48,48))