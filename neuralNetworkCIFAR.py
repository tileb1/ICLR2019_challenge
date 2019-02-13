import torch
import torch.nn as nn
import torch.nn.functional as F

class CIFAR10_NET(nn.Module):

    def __init__(self):
        super(CIFAR10_NET, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1),
            nn.ReLU())
        self.conv4 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.conv5 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.conv6 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=1),
            nn.ReLU())
        self.conv7 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.conv8 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=1, stride=1),
            nn.ReLU())
        self.conv9 = nn.Sequential(
            nn.Conv2d(192, 10, kernel_size=1, stride=1),
            nn.ReLU())
        self.avgPool1 = nn.AvgPool2d(kernel_size=6, stride=1)

        self.fc1 = nn.Linear(3 * 3 * 10, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.avgPool1(x)
        x = x.view(-1, 90)
        x = self.fc1(x)
        return F.softmax(x, dim=1)
