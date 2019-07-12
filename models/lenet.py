import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class Square(nn.Module):
    def __init__(self):
        super(Square, self).__init__()

    def forward(self, x):
        return torch.mul(x, x)


class CryptoNet(nn.Module):
    def __init__(self):
        super(CryptoNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=5, stride=2, padding=0)
        self.relu1 = nn.ReLU()
        #self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(720, 100)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = x.view(-1, 180)
        x = self.relu2(self.fc1(x))
        x = self.fc2(x)
        return x

class SquareLeNet(nn.Module):
    def __init__(self):
        super(SquareLeNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            Square(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            Square(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            Square(),
            nn.Linear(120, 84),
            Square(),
            nn.Linear(84, 10),
        )

        for m in self.features.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.classifier.children():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    import torch
    x = torch.ones((1, 1, 28, 28))
    net = CryptoNet()
    y = net(x)