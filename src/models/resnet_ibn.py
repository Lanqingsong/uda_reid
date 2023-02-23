from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
from .resnet_ibn_a import resnet50_ibn_a, resnet101_ibn_a
import torch

__all__ = ['ResNetIBN', 'resnet_ibn50a', 'resnet_ibn101a']


class ResNetIBN(nn.Module):
    __factory = {
        '50a': resnet50_ibn_a,
        '101a': resnet101_ibn_a
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0):
        super(ResNetIBN, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        resnet = ResNetIBN.__factory[depth](pretrained=pretrained)
        resnet.layer4[0].conv2.stride = (1,1)
        resnet.layer4[0].downsample[0].stride = (1,1)

        self.base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)


        # self.base0 =  nn.Sequential( resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        # self.base1 = nn.Sequential( resnet.layer1)
        # self.base2 = nn.Sequential( resnet.layer2)
        # self.base3 = nn.Sequential( resnet.layer3)
        # self.base4 = nn.Sequential( resnet.layer4)

        self.gap = nn.AdaptiveAvgPool2d(1)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = resnet.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
                self.feat_bn = nn.BatchNorm1d(self.num_features)
            self.feat_bn.bias.requires_grad_(False)
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
                init.normal_(self.classifier.weight, std=0.001)
        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)


        if not pretrained:
            self.reset_params()

#         for param in self.parameters():
#             param.requires_grad = False
        self.conv_bottle = nn.Conv2d(2048, 1024, kernel_size=(3,3),padding=1,bias=False)
        self.convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=512,  output_padding=(1,1),kernel_size=(3,3), stride=(2,2),
                               padding=(1,1)),
            nn.BatchNorm2d(512, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256,  output_padding=(1,1),kernel_size=(3,3), stride=(2,2),
                               padding=(1,1)),
            nn.BatchNorm2d(256, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=64,  output_padding=(1,1),kernel_size=(3,3), stride=(2,2),
                               padding=(1,1)),
            nn.BatchNorm2d(64, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.convTrans9 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=3,  output_padding=(1,1),kernel_size=(5,5), stride=(2,2),
                               padding=(2,2)),
            # nn.BatchNorm2d(64, momentum=0.01),
            # nn.ReLU(inplace=True),
        )


    def forward(self, x):
        # x = self.base0(x)#32,64,64,32
        # x = self.base1(x)#32,256,64,32
        # x = self.base2(x)#32,512,32,16
        # x = self.base3(x)#32,1024,16,8
        # x = self.base4(x)#32,2048,16,8
        inputs = x
#         with torch.no_grad():
        x = self.base(x)
        ##todo
        y =self.conv_bottle(x)#1054,16,8
        y =  self.convTrans6(y) #512, 32 16
        y =  self.convTrans7(y)#256, 64 32
        y = self.convTrans8(y)
        y = self.convTrans9(y)#32,3,256,128

        y=F.mse_loss(y,inputs)

        ##

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        if self.cut_at_pooling:
            return x

        if self.has_embedding:
            bn_x = self.feat_bn(self.feat(x))
        else:
            bn_x = self.feat_bn(x)

        if self.training is False:
            bn_x = F.normalize(bn_x)
            return bn_x

        if self.norm:
            bn_x = F.normalize(bn_x)
        elif self.has_embedding:
            bn_x = F.relu(bn_x)

        if self.dropout > 0:
            bn_x = self.drop(bn_x)

        if self.num_classes > 0:
            prob = self.classifier(bn_x)
        else:
            return bn_x, y

        return prob, x, y

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


def resnet_ibn50a(**kwargs):
    return ResNetIBN('50a', **kwargs)


def resnet_ibn101a(**kwargs):
    return ResNetIBN('101a', **kwargs)
