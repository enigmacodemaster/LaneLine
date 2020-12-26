import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):

    def __init__(self, in_chans, out_chans, rate=1): # rate 膨胀率比例
        super(ASPP, self).__init__() # 父类
        # your code
        # ...
        self.branch1 = nn.Sequential(
        		nn.Conv2d(in_chans, out_chans, 1, 1, padding=0, dilation=rate, bias=True),
        		nn.BatchNorm2d(out_chans),
        		nn.Relu(inplace=True)
        	)
       	self.branch2 = nn.Sequential(
       			nn.Conv2d(in_chans, out_chans, 3, 1, padding=6*rate, dilation=rate * 6, bias=True),
       			nn.BatchNorm2d(out_chans),
       			nn.Relu(inplace=True)
       		)
       	self.branch3 = nn.Sequential(
       			nn.Conv2d(in_chans, out_chans, 3, 1, padding=12*rate, dilation=rate * 12, bias=True),
       			nn.BatchNorm2d(out_chans),
       			nn.Relu(inplace=True)
       		)
       	self.branch4 = nn.Sequential(
       			nn.Conv2d(in_chans, out_chans, 3, 1, padding=18*rate, dilation=rate * 18, bias=True),
       			nn.BatchNorm2d(out_chans),
       			nn.Relu(inplace=True),
       		)
       	self.avgPooling = nn.AdaptiveAbgPool2d(1) # 自适应的pooling成对应的feature map大小，调整到1x1的feature map
       	self.branch5 = nn.Sequential( # 调整1x1 feature map的通道数
       			nn.Conv2d(in_chans, out_chans, 1, 1, 0, bias=True),
       			nn.BatchNorm2d(out_chans),
       			nn.Relu(inplace=True),
       		)
       	self.conv = nn.Sequential(
       			nn.Conv2d(out_chans * 5, out_chans, 1, 1, padding = 0, bias = True),
       			nn.BatchNorm2d(out_chans),
       			nn.Relu(inplace=True)
       		)


    def forward(self, x):
        # your code
        # ...
        b, c, h, w = x.size()
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        global_feature = self.avgPooling(x)
        global_feature = self.branch5(global_feature)
        global_feature = F.interpolate(global_feature, (h, w), None, 'bilinear', True)

        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv(feature_cat)

        return result