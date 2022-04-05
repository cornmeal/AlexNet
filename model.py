from torch import nn
from torch.nn import functional as F


# AlexNet模型,继承自nn.Module
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.fc6 = nn.Linear(9216, 4096)
        self.fc7 = nn.Linear(4096, 1000)
        self.fc8 = nn.Linear(1000, 11)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(3, stride=2)
        self.pool2 = nn.MaxPool2d(3, stride=2)
        self.drop = nn.Dropout(0.5)
        self.lrn = nn.LocalResponseNorm(size=5, alpha=10e-4, beta=0.75, k=2.0)

    def forward(self, x):
        x = self.pool1(self.lrn(F.relu(self.conv1(x))))
        x = self.pool2(self.lrn(F.relu(self.conv2(x))))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(F.relu(self.conv5(x)))
        # 相当于reshape函数，展平也可以用torch.flatten(x, dim=1)
        x = x.view(-1, 9216)
        x = F.relu(self.drop(self.fc6(x)))
        x = F.relu(self.drop(self.fc7(x)))
        x = self.fc8(x)
        return x
