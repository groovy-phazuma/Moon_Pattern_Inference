# -*- coding: utf-8 -*-
"""
# training using ImageNet-S

@author: Katsuhisa MORITA
"""
# import
import os
import sys
import datetime
import random
import logging
import datetime
import argparse
import time
import random
from typing import List, Tuple, Union, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

from PIL import ImageOps, Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from timm.scheduler import CosineLRScheduler

# path settings
PROJECT_PATH = '/workspace/Moon_Pattern_Inference'
DIR_IMAGENETS = "/workspace/ImageNet/ImageNet-S/ImageNetS919"

# impor module
sys.path.append(PROJECT_PATH)
import silhouette_model.utils as utils
import silhouette_model.models as models

# argument
parser = argparse.ArgumentParser(description='CLI learning')
# base settings
parser.add_argument('--note', type=str, help='ImageNet-S classification')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dir_result', type=str, help='ViT')
# model/learning settings
parser.add_argument('--model_name', type=str, default='ViT') # model architecture name
parser.add_argument('--num_epoch', type=int, default=100) # epoch
parser.add_argument('--batch_size', type=int, default=128) # batch size
parser.add_argument('--lr', type=float, default=1e-3) # learning rate
parser.add_argument('--patience', type=int, default=5) # early stopping
parser.add_argument('--delta', type=float, default=0.002) # early stopping
# scheduler
parser.add_argument('--lr_min', type=float, default=1e-6)
parser.add_argument('--warmup_t', type=int, default=10)
parser.add_argument('--warmup_lr_init', type=float, default=1e-6)
# parse
args = parser.parse_args()
# fix seed
utils.fix_seed(seed=args.seed, fix_gpu=True)

# Other parameters
DICT_MODEL={
    "ViT": models.ViT,
    "SwinB": models.SwinB,
    "ConvNext": models.ConvNext,
    "ResNet50": models.ResNet50,
}
NUM_CLASSES=919

# utils
def set_transforms():
    # normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    # for training dataset
    train_data_transform = transforms.Compose([
        transforms.Resize((256,256)),
        utils.Transforms_Segmentation(),
        transforms.RandomHorizontalFlip(p=0.5),
        utils.random_rotation_transform(rr_prob=1., rr_degrees=[0,180]),
        #transforms.RandomApply([
        #    transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8
        #    ), # don't need color jitter for silhouette
        #transforms.RandomGrayscale(p=0.2), # don't need grey scaling for silhouette
        #transforms.RandomApply([
        #    transforms.GaussianBlur((3, 3), (1.0, 2.0))], p=0.2
        #    ), # don't need blur for silhouette
        transforms.RandomResizedCrop((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    # for validation dataset
    val_data_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        normalize
    ])
    return train_data_transform, val_data_transform

def prep_dataloader(
    dataset, batch_size:int, shuffle:bool=True, num_workers:int=4, pin_memory:bool=True, drop_last:bool=True, sampler=None
    ) -> torch.utils.data.DataLoader:
    """
    prepare train and test loader
    
    Parameters
    ----------
    dataset: torch.utils.data.Dataset
        prepared Dataset instance
    
    batch_size: int
        the batch size
    
    shuffle: bool
        whether data is shuffled or not

    num_workers: int
        the number of threads or cores for computing
        should be greater than 2 for fast computing
    
    pin_memory: bool
        determines use of memory pinning
        should be True for fast computing
    
    """
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_worker_init_fn,
        drop_last=drop_last,
        sampler=sampler,
        )
    return loader

def _worker_init_fn(worker_id):
    """ fix the seed for each worker """
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def prep_train_dataloader(batch_size:int=128):
    """ datasets for training"""
    train_data_transform=set_transforms()[0]
    dataset = torchvision.datasets.ImageFolder(f"{DIR_IMAGENETS}/train-semi-segmentation", transform=train_data_transform)
    return prep_dataloader(
        dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=4, 
        pin_memory=True, 
        drop_last=True, 
        sampler=None
        )

def prep_val_dataloader(batch_size:int=128):
    """ datasets for validation"""
    val_data_transform=set_transforms()[1]
    dataset = torchvision.datasets.ImageFolder(f"{DIR_IMAGENETS}/validation-segmentation", transform=val_data_transform)
    return prep_dataloader(
        dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=4, 
        pin_memory=True, 
        drop_last=False, 
        sampler=None
        )

def prepare_model(
    model_name:str='', 
    patience:int=7, 
    delta:float=0, 
    lr:float=0.003,
    num_epoch:int=150,
    lr_min:float=0.00001,
    warmup_t:int=5,
    warmup_lr_init:float=0.00001,
    ):
    model = DICT_MODEL[model_name](num_classes=NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineLRScheduler(
        optimizer, t_initial=num_epoch, lr_min=lr_min,
        warmup_t=warmup_t, warmup_lr_init=warmup_lr_init, warmup_prefix=True)
    early_stopping = utils.EarlyStopping(patience=patience, delta=delta, path=f'{DIR_NAME}/checkpoint.pt')
    model.to(DEVICE)
    return model, criterion, optimizer, scheduler, early_stopping

# Training
def train_epoch(model, criterion, optimizer, train_dataloader, val_dataloader):
    """model training for epoch"""
    train_batch_loss=[]
    val_batch_loss=[]
    train_batch_acc=[]
    val_batch_acc=[]
    # training
    model.train()
    for data, label in train_dataloader:
        data, label = data.to(DEVICE), label.to(DEVICE)
        output=model(data)
        loss=criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # append loss and acc
        acc = (output.argmax(dim=1) == label).float().mean()
        train_batch_loss.append(loss.item())
        train_batch_acc.append(acc.item())
    # validation
    model.eval()
    with torch.inference_mode():
        for data, label in val_dataloader:
            data, label = data.to(DEVICE), label.to(DEVICE)
            output=model(data)
            loss=criterion(output, label)
            acc = (output.argmax(dim=1) == label).float().mean()
            val_batch_loss.append(loss.item())
            val_batch_acc.append(acc.item())

    return model, np.mean(train_batch_loss), np.mean(train_batch_acc), np.mean(val_batch_loss), np.mean(val_batch_acc)

def train(model, criterion, optimizer, scheduler, early_stopping, num_epoch:int=100, batch_size:int=128):
    """model training"""
    # settings
    start=time.time()
    train_dataloader=prep_train_dataloader(batch_size=batch_size)
    val_dataloader=prep_val_dataloader(batch_size=batch_size)
    train_loss=[]
    train_acc=[]
    val_loss=[]
    val_acc=[]
    for epoch in range(num_epoch):
        # training
        model, train_epoch_loss, train_epoch_acc, val_epoch_loss, val_epoch_acc = train_epoch(
            model, criterion, optimizer,
            train_dataloader, val_dataloader
            )
        train_loss.append(train_epoch_loss)
        train_acc.append(train_epoch_acc)
        val_loss.append(val_epoch_loss)
        val_acc.append(val_epoch_acc)
        scheduler.step(epoch)
        LOGGER.logger.info(
            f'Epoch: {epoch + 1}, train_loss: {train_epoch_loss:.4f}, val_loss: {val_epoch_loss:.4f}, train_acc: {train_epoch_acc:.4f}, val_acc: {val_epoch_acc:.4f}'
            )
        LOGGER.logger.info('elapsed_time: {:.2f} min'.format((time.time() - start)/60))
        # early stopping
        early_stopping(val_epoch_loss, model)
        if early_stopping.early_stop:
            LOGGER.logger.info(f'Early Stopping with Epoch: {epoch}')
            model.load_state_dict(torch.load(early_stopping.path))        
            return model, train_loss, val_loss
    return model, train_loss, train_acc, val_loss, val_acc

def main():
    # prepare models
    model, criterion, optimizer, scheduler, early_stopping = prepare_model(
        model_name=args.model_name, patience=args.patience, delta=args.delta, lr=args.lr, num_epoch=args.num_epoch,
        lr_min=args.lr_min, warmup_t=args.warmup_t, warmup_lr_init=args.warmup_lr_init,
    )
    # training
    model, train_loss, train_acc, val_loss, val_acc = train(
        model, criterion, optimizer, scheduler, early_stopping, num_epoch=args.num_epoch
    )
    # plot, save model
    utils.plot_progress(train_loss, val_loss, outdir=DIR_NAME, name="loss", label="loss")
    utils.plot_progress(train_loss, val_loss, outdir=DIR_NAME, name="acc", label="accuracy")
    torch.save(model.state_dict(), f'{DIR_NAME}/model.pt')
    # logging
    LOGGER.to_logger(name='argument', obj=args)
    LOGGER.to_logger(name='loss', obj=criterion)
    LOGGER.to_logger(
        name='optimizer', obj=optimizer, skip_keys={'state', 'param_groups'}
    )
    LOGGER.to_logger(name='scheduler', obj=scheduler)

if __name__=='__main__':
    filename = os.path.basename(__file__).split('.')[0]
    DIR_NAME = PROJECT_PATH + '/result/' +args.dir_result # for output
    DEVICE = torch.device('cuda:0') # if torch.cuda.is_available() else 'cpu') # get device
    if not os.path.exists(DIR_NAME):
        os.makedirs(DIR_NAME)
    now = datetime.datetime.now().strftime('%H%M%S')
    LOGGER = utils.logger_save()
    LOGGER.init_logger(filename, DIR_NAME, now, level_console='debug') 
    main()