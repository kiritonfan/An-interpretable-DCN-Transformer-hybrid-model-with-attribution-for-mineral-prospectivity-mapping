"""This module defines the DCNv2-based feature extraction components used in the study."""

import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d


class DCNv2Block(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1,
                 dilation=1, deformable_groups=1):
        super(DCNv2Block, self).__init__()

                                           
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size * kernel_size * deformable_groups,
            kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=True
        )

                                                      
        self.mask_conv = nn.Conv2d(
            in_channels,
            kernel_size * kernel_size * deformable_groups,
            kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=True
        )

                                      
        self.dcn = DeformConv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=False
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

                                         
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)
        nn.init.constant_(self.mask_conv.weight, 0)
        nn.init.constant_(self.mask_conv.bias, 0)

    def forward(self, x):
        offset = self.offset_conv(x)                                 
        mask = torch.sigmoid(self.mask_conv(x))                            
        x = self.dcn(x, offset, mask)                                 
        x = self.bn(x)
        x = self.relu(x)
        return x


class DCNv2(nn.Module):

    def __init__(self, num_classes=2, output_dim=512):
        super(DCNv2, self).__init__()

        self.output_dim = output_dim
        self.gradients = None
        self.activations = None

                             
        self.block1 = nn.Sequential(
            nn.Conv2d(42, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

                                    
        self.block2 = DCNv2Block(64, 128, kernel_size=3, padding=1)

                                                 
        self.block3 = nn.Conv2d(128, output_dim, kernel_size=1)

                      
        hidden_dim = max(64, output_dim // 2)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x, extract_features=False):
        x = self.block1(x)                         
        x = self.block2(x)                         
        features = self.block3(x)                         

        self.activations = features
        if features.requires_grad:
            features.register_hook(self.activations_hook)

        if extract_features:
            return features

        return self.classifier(features)

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self):
        return self.activations
