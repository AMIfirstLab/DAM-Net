import torch
import torch.nn as nn


class SE_Block(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y)
        y = y.view(bs, c, 1, 1)
        return x * y.expand_as(x)


class Conv_2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Conv_2d, self).__init__()
        
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, padding = 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
    
    def forward(self, x):
        x = self.conv_layer(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class Convolution_Block(nn.Module):
    def __init__(self, in_channels, start_channels):
        super().__init__()
        
        self.pool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        self.conv1 = Conv_2d(in_channels, start_channels, kernel_size = 3)
        self.conv2 = Conv_2d(start_channels, start_channels, kernel_size = 2)
        
        self.conv3 = Conv_2d(start_channels, int(start_channels*2), kernel_size = 3)
        self.conv4 = Conv_2d(int(start_channels*2), int(start_channels*2), kernel_size = 3)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.conv4(x)
        
        return x

class Dense_layer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv0 = Conv_2d(in_channels, in_channels, kernel_size = 3)
        self.conv1 = Conv_2d(in_channels, in_channels, kernel_size = 3)
        self.conv2 = Conv_2d(in_channels, in_channels, kernel_size = 3)
        self.conv3 = Conv_2d(in_channels, in_channels, kernel_size = 3)
        self.conv4 = Conv_2d(in_channels, in_channels, kernel_size = 3)
        
    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1 + x0)
        x3 = self.conv3(x2 + x1 + x0)
        x4 = self.conv4(x3 + x2 + x1 + x0)
        
        return x4

class Transitional_layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.avgpool = nn.AvgPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size = 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
        
    def forward(self, x):
        x = self.conv_layer(x)
        x = self.bn(x)
        x = self.act(x)
        
        x = self.avgpool(x)
        
        return x

class Linear_layer(nn.Module):
    def __init__(self, classes):
        super().__init__()
        
        self.linear_1 = nn.Linear(256*9*9, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.act_1 = nn.ReLU()
        
        self.linear_2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.act_2 = nn.ReLU()
        
        self.linear_3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.act_3 = nn.ReLU()
        
        self.linear_4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.act_4 = nn.ReLU()
        
        self.linear_5 = nn.Linear(32, classes)
        self.bn5 = nn.BatchNorm1d(classes)
        self.act_5 = nn.Softmax(dim = 1)
        
    
    def forward(self, x):
        x = torch.flatten(x, start_dim = 1)
        x = self.act_1(self.bn1(self.linear_1(x)))
        x = self.act_2(self.bn2(self.linear_2(x)))
        x = self.act_3(self.bn3(self.linear_3(x)))
        x = self.act_4(self.bn4(self.linear_4(x)))
        x = self.act_5(self.bn5(self.linear_5(x)))
        
        return x

class DAM_NET(nn.Module):
    def __init__(self, in_channels, classes):
        super().__init__()
        
        self.conv_block = Convolution_Block(in_channels, 32)
        
        self.se_1 = SE_Block(32)
        self.dense_1 = Dense_layer(32)
        self.se_12 = SE_Block(32)
        
        self.trans_1 = Transitional_layer(32, 64)
        
        self.se_2 = SE_Block(64)
        self.dense_2 = Dense_layer(64)
        self.se_22 = SE_Block(64)
        
        self.trans_2 = Transitional_layer(64, 128)
        
        self.se_3 = SE_Block(128)
        self.dense_3 = Dense_layer(128)
        self.se_32 = SE_Block(128)
        
        self.trans_3 = Transitional_layer(128, 256)
        
        self.se_4 = SE_Block(256)
        self.dense_4 = Dense_layer(256)
        
        self.avgpool = nn.AvgPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        self.linear_layers = Linear_layer(classes)
        
    
    def forward(self, x):
        x = self.conv_block(x)
        
        x = self.se_1(x)
        x = self.dense_1(x)
        x = self.se_12(x)
        
        x = self.trans_1(x)
        
        x = self.se_2(x)
        x = self.dense_2(x)
        x = self.se_22(x)
        
        x = self.trans_2(x)
        
        x = self.se_3(x)
        x = self.dense_3(x)
        x = self.se_32(x)
        
        x = self.trans_3(x)
        x4 = self.se_4(x)
        x = self.dense_4(x4)
        
        x = self.avgpool(x)
        
        x = self.linear_layers(x)
        
        return x
