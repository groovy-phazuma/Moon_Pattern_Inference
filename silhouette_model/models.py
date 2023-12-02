# -*- coding: utf-8 -*-
"""
# models for classification (pre-trained transfer)

@author: Katsuhisa MORITA
"""

import torch
import torch.nn as nn
import torchvision
import timm

def ViT(num_classes):
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
    return model

def SwinB(num_classes):
    model = timm.create_model('swin_s3_base_224', pretrained=True, num_classes=num_classes)
    return model
    
def ResNet50(num_classes):
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def ConvNext(num_classes):
    model = torchvision.models.convnext_base(weights=torchvision.models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
    model.classifier[2] = nn.Linear(model.lastconv_output_channels, num_classes)
    return model