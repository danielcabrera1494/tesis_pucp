import torch.nn as nn
import torch
import numpy as np


class StutterNet_1(nn.Module):
    def __init__(self, batch_size):
        super(StutterNet, self).__init__()
        # input shape = (batch_size, 1, 149,1024)
        # in_channels is batch size
        self.layer1 = nn.Sequential(
            torch.nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=0.5)
        )
        self.layer1_bn = nn.BatchNorm2d(8)
        # input size = (batch_size, 8, 74, 384)
        self.layer2 = nn.Sequential(
            torch.nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=1, stride=2),
            torch.nn.Dropout(p=0.5)
        )

        self.layer3 = nn.Sequential(
            torch.nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=1, stride=2),
            torch.nn.Dropout(p=0.5)
        )
        self.layer4 = nn.Sequential(
            torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=1, stride=2),
            torch.nn.Dropout(p=0.5)
        )
        self.layer2_bn = nn.BatchNorm2d(16)
        self.layer3_bn = nn.BatchNorm2d(32)
        self.layer4_bn = nn.BatchNorm2d(64)
        # input size = (batch_size, 16, 37, 192)
        self.flatten = torch.nn.Flatten()
        self.fc1 = nn.Linear(16* 37* 256,512, bias=True)
        self.fc1_bn = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512,256, bias=True)
        self.fc2_bn = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256,128, bias=True)
        self.fc3_bn = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128,64, bias=True) #32 to 64
        self.fc4_bn = nn.BatchNorm1d(64)

        self.fc6 = nn.Linear(64,32, bias=True)
        self.fc6_bn = nn.BatchNorm1d(32)

        self.fc5 = nn.Linear(32,2, bias=True)

        self.relu = nn.LeakyReLU()
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        #print('Before Layer1',np.shape(x))
        out = self.layer1(x)
        # out = self.layer1_bn(out)
        # print('After layer 1',np.shape(out))
        out = self.layer2(out)
        # out = self.layer2_bn(out)
        # print('After layer 2',np.shape(out))
        out = self.layer3(out)
        #print('After layer 3',np.shape(out))

        out = self.layer4(out)
        print('After layer 3',np.shape(out))

        out  = self.flatten(out)

        out = self.fc1(out)
        out = self.relu(out)
        # out = self.fc1_bn(out)

        out = self.fc2(out)
        out = self.relu(out)
        # out = self.fc2_bn(out)

        out = self.fc3(out)
        out = self.relu(out)
        # out = self.fc3_bn(out)

        out = self.fc4(out)
        out = self.relu(out)
        # out = self.fc4_bn(out)

        out = self.fc5(out)
        out = self.sm(out)
        #print('After final ',np.shape(out))

        # log_probs = torch.nn.functional.log_softmax(out, dim=1)

        return out

class StutterNet(nn.Module):
    def __init__(self, batch_size):
        super(StutterNet, self).__init__()
        # input shape = (batch_size, 1, 149,768)
        # in_channels is batch size
        self.layer1 = nn.Sequential(
            torch.nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
<<<<<<< Updated upstream
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
=======
            torch.nn.MaxPool2d(kernel_size=2, stride=1),
>>>>>>> Stashed changes
            torch.nn.Dropout(p=0.5)
        )
        self.layer1_bn = nn.BatchNorm2d(8)
        # input size = (batch_size, 8, 74, 384)
        self.layer2 = nn.Sequential(
            torch.nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
<<<<<<< Updated upstream
            torch.nn.MaxPool2d(kernel_size=1, stride=2),
=======
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
>>>>>>> Stashed changes
            torch.nn.Dropout(p=0.5)
        )
        self.layer2_bn = nn.BatchNorm2d(16)

        self.layer3 = nn.Sequential(
            torch.nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=1, stride=2),
            torch.nn.Dropout(p=0.5)
        )

        # input size = (batch_size, 16, 37, 192)
        self.flatten = torch.nn.Flatten()
<<<<<<< Updated upstream
        self.fc1 = nn.Linear(16* 37* 256,500, bias=True)
=======
        self.fc1 = nn.Linear(32*37*256, 500, bias=True)
>>>>>>> Stashed changes
        self.fc1_bn = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500,250, bias=True)
        self.fc2_bn = nn.BatchNorm1d(250)
        self.fc3 = nn.Linear(250,100, bias=True)
        self.fc3_bn = nn.BatchNorm1d(100)
        self.fc4 = nn.Linear(100,10, bias=True)
        self.fc4_bn = nn.BatchNorm1d(10)
        #self.fc5 = nn.Linear(10,2, bias=True)
        self.fc5 = nn.Linear(10,1) #, bias=True)

        self.relu = nn.LeakyReLU()
        self.sm = nn.Softmax()

    def forward(self, x):
        #print('Before Layer1',np.shape(x))
        out = self.layer1(x)
        # out = self.layer1_bn(out)
        #print('After layer 1',np.shape(out))
        out = self.layer2(out)
        # out = self.layer2_bn(out)
        #print('After layer 2',np.shape(out))

        out = self.layer3(out)
        print('After layer 3',np.shape(out))

        out  = self.flatten(out)

        out = self.fc1(out)
        out = self.relu(out)
        # out = self.fc1_bn(out)

        out = self.fc2(out)
        out = self.relu(out)
        # out = self.fc2_bn(out)

        out = self.fc3(out)
        out = self.relu(out)
        # out = self.fc3_bn(out)

        out = self.fc4(out)
        out = self.relu(out)
        # out = self.fc4_bn(out)

        out = self.fc5(out)
        #out = self.sm(out)
        #print('After final ',np.shape(out))

        # log_probs = torch.nn.functional.log_softmax(out, dim=1)
        return out

