import os 
import torch
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
from PIL import Image
from torch.nn import init
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # 正常卷积部分，堆叠了两层卷积
        self.left = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), # 原地替换 节省内存开销
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # 如果上方卷积没有改变size和channel
        # 则不需要对输入进行变化，故shortcut为空
        self.shortcut = nn.Sequential()
        # 如果上方卷积改变了size和channel
        # 则使用1×1卷积改变输入的size和channel，使其保持一致
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                # 卷积核为1 进行升降维
                # 注意跳变时 都是stride==2的时候
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, X):
        out = self.left(X)
        out += self.shortcut(X)  #将输入/变化shape后的输入与卷积的输出相加
        out = F.relu(out)  # 经过激活函数后输出，注意先相加再激活
        return out
    
    

class MyResNet(nn.Module):
    def __init__(self, num_classes=3) -> None:
        super(MyResNet, self).__init__()
        self.in_channels = 64
        # 第一层作为单独的 因为没有残差快
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )
        
        # 按照图像构造对应残差连接
        self.conv2 = self._make_layer(64,[[1,1],[1,1]])
        self.conv3 = self._make_layer(128,[[2,2],[1,1]])
        self.conv4 = self._make_layer(256,[[2,2],[1,1]])
        self.conv5 = self._make_layer(512,[[2,2],[1,1]])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))   
        # 该处使用三个全连接层
        self.fc = nn.Sequential(
                nn.Linear(512, 256),
                nn.Linear(256, 128),
                nn.Linear(128, num_classes)
        )

    #构建重复的残差块
    def _make_layer(self, out_channels, strides):
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    # 图片的大小为 64*64
    def forward(self, x):
        out = self.conv1(x)   # [32, 64, 64, 64]
        out = self.conv2(out)  # [32, 64, 64, 64]
        out = self.conv3(out)  # [32, 128, 32, 32]
        out = self.conv4(out)  # [32, 256, 16, 16]
        out = self.conv5(out)  # [32, 512, 8, 8]
        out = F.avg_pool2d(out,8)  # [32, 512, 1, 1]
        out = out.squeeze()  # [32, 512]
        out = self.fc(out)
        return out
    
    
data = torch.rand(4,3,64,64)

model = MyResNet()
res = model(data)
print(res.shape)