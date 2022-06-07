import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import sys
import math

from config import get_args

global_args = get_args(sys.argv[1:])


class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化,输入BCHW -> 输出 B*C*1*1
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),  # 可以看到channel得被reduction整除，否则可能出问题
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # squeeze操作,得到B*C*1*1,然后转成B*C，才能送入到FC层中。
        y = self.fc(y).view(b, c, 1, 1)  # FC获取通道注意力权重，是具有全局信息的,得到B*C的向量，C个值就表示C个通道的权重。把B*C变为B*C*1*1是为了与四维的x运算。
        feature = x * y.expand_as(x)  # 注意力作用每一个通道上,先把B*C*1*1变成B*C*H*W大小，其中每个通道上的H*W个值都相等。*表示对应位置相乘。
        return feature


class DenseAsppBlock(nn.Sequential):
    """ ConvNet block for building DenseASPP. """

    def __init__(self, input_num, num1, num2, dilation_rate, drop_out, bn_start=True):
        super(DenseAsppBlock, self).__init__()
        if bn_start:
            self.add_module('norm_1', nn.BatchNorm2d(input_num, momentum=0.0003)),

        self.add_module('relu_1', nn.ReLU(inplace=True)),
        self.add_module('conv_1', nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1)),

        self.add_module('norm_2', nn.BatchNorm2d(num1, momentum=0.0003)),
        self.add_module('relu_2', nn.ReLU(inplace=True)),
        self.add_module('conv_2', nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
                                            dilation=dilation_rate, padding=dilation_rate)),

        self.drop_rate = drop_out

    def forward(self, _input):
        feature = super(DenseAsppBlock, self).forward(_input)

        if self.drop_rate > 0:
            feature = F.dropout2d(feature, p=self.drop_rate, training=self.training)
        return feature


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def get_sinusoid_encoding(n_position, feat_dim, wave_length=10000):
    # [n_position]
    positions = torch.arange(0, n_position)  # .cuda()
    # [feat_dim]
    dim_range = torch.arange(0, feat_dim)  # .cuda()
    dim_range = torch.pow(wave_length, 2 * (dim_range // 2) / feat_dim)
    # [n_position, feat_dim]
    angles = positions.unsqueeze(1) / dim_range.unsqueeze(0)
    angles = angles.float()
    angles[:, 0::2] = torch.sin(angles[:, 0::2])
    angles[:, 1::2] = torch.cos(angles[:, 1::2])
    return angles


class AsterBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(AsterBlock, self).__init__()
        self.conv1 = conv1x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SE_Block(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet_ASTER(nn.Module):
    """For aster or crnn"""

    def __init__(self, with_lstm=False, n_group=1):
        super(ResNet_ASTER, self).__init__()
        self.with_lstm = True
        self.n_group = n_group

        in_channels = 3
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))

        self.inplanes = 32
        self.layer1 = self._make_layer(32, 3, [2, 2])  # [16, 50]
        self.layer2 = self._make_layer(64, 4, [2, 2])  # [8, 25]
        self.layer3 = self._make_layer(128, 6, [2, 1])  # [4, 25]
        self.layer4 = self._make_layer(256, 6, [2, 1])  # [2, 25]
        self.layer5 = self._make_layer(512, 3, [2, 1])  # [1, 25]

        num_features = 512
        d_feature0 = 512
        d_feature1 = 512
        dropout0 = 0.2

        self.ASPP_3 = DenseAsppBlock(input_num=num_features, num1=d_feature0, num2=d_feature1,
                                     dilation_rate=3, drop_out=dropout0, bn_start=False)

        self.ASPP_6 = DenseAsppBlock(input_num=num_features + d_feature1 * 1, num1=d_feature0, num2=d_feature1,
                                     dilation_rate=6, drop_out=dropout0, bn_start=True)

        self.ASPP_12 = DenseAsppBlock(input_num=num_features + d_feature1 * 2, num1=d_feature0, num2=d_feature1,
                                      dilation_rate=12, drop_out=dropout0, bn_start=True)

        self.ASPP_18 = DenseAsppBlock(input_num=num_features + d_feature1 * 3, num1=d_feature0, num2=d_feature1,
                                      dilation_rate=18, drop_out=dropout0, bn_start=True)

        self.ASPP_24 = DenseAsppBlock(input_num=num_features + d_feature1 * 4, num1=d_feature0, num2=d_feature1,
                                      dilation_rate=24, drop_out=dropout0, bn_start=True)

        self.conv_1x1_output = nn.Conv2d(d_feature1 * 6, d_feature1, 1, 1)

        if self.with_lstm:
            self.rnn = nn.LSTM(512, 256, bidirectional=True, num_layers=2, batch_first=True)
            self.out_planes = 2 * 256
            print('use lstm')
        else:
            self.out_planes = 512
            print('not use lstm')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride):
        downsample = None
        if stride != [1, 1] or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(AsterBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(AsterBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        feature = x5
        aspp3 = self.ASPP_3(feature)
        feature = torch.cat((aspp3, feature), dim=1)
        aspp6 = self.ASPP_6(feature)
        feature = torch.cat((aspp6, feature), dim=1)
        aspp12 = self.ASPP_12(feature)
        feature = torch.cat((aspp12, feature), dim=1)
        aspp18 = self.ASPP_18(feature)
        feature = torch.cat((aspp18, feature), dim=1)
        aspp24 = self.ASPP_24(feature)
        feature = torch.cat((aspp24, feature), dim=1)
        feature = self.conv_1x1_output(feature)
        cnn_feat = feature.squeeze(2)  # [N, c, w]
        cnn_feat = cnn_feat.transpose(2, 1)
        if self.with_lstm:
            rnn_feat, _ = self.rnn(cnn_feat)
            return rnn_feat
        else:
            return cnn_feat


if __name__ == "__main__":
    x = torch.randn(3, 3, 32, 100)
    net = ResNet_ASTER(use_self_attention=True, use_position_embedding=True)
    encoder_feat = net(x)
    print(encoder_feat.size())
