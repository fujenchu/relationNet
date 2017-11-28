import torch.nn as nn
import torch.nn.functional as F

'''
define networks
'''
class Net(nn.Module):
    def __init__(self, num_in_channel, num_filter):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(num_in_channel, num_filter, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filter)
        self.conv2 = nn.Conv2d(num_filter, num_filter, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filter)
        self.conv3 = nn.Conv2d(num_filter, num_filter, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_filter)
        self.conv4 = nn.Conv2d(num_filter, num_filter, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(num_filter)
        self.pool = nn.MaxPool2d(2, stride=2)


    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x)))) # 40x40
        x = self.pool(F.relu(self.bn2(self.conv2(x)))) # 20x20
        x = F.relu(self.bn3(self.conv3(x))) # 20x20
        x = F.relu(self.bn4(self.conv4(x))) # bx64x20x20
        return x


class RelationNet(nn.Module):
    def __init__(self, num_in_channel, num_filter, num_fc1, num_fc2):
        super(RelationNet, self).__init__()
        self.conv1 = nn.Conv2d(num_in_channel, num_filter, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filter)
        self.conv2 = nn.Conv2d(num_filter, num_filter, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filter)
        self.fc1 = nn.Linear(num_fc1, num_fc2)
        self.fc2 = nn.Linear(num_fc2, 1)
        self.pool = nn.MaxPool2d(2, stride=2)


    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x)))) # 10x10
        x = self.pool(F.relu(self.bn2(self.conv2(x)))) # 5x5
        x = x.view(x.size()[0], -1) # 6400
        x = F.relu(self.fc1(x)) #8
        x = F.sigmoid(self.fc2(x)) #1
        return x