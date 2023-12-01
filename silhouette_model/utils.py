# -*- coding: utf-8 -*-
"""
# training utils using ImageNet-S

@author: Katsuhisa MORITA
"""

import os
import datetime
import random
import logging
from typing import List, Tuple, Union, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor

import matplotlib.pyplot as plt
from sklearn import metrics
from PIL import ImageOps, Image
import torchvision.transforms as transforms

# assist model building
def fix_seed(seed:int=None,fix_gpu:bool=True):
    """ fix seed """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if fix_gpu:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def fix_params(model, forall=False):
    """ freeze model parameters """
    # freeze layers
    for param in model.parameters():
        param.requires_grad = False
    # except last layer
    if forall:
        pass
    else:
        last_layer = list(model.children())[-1]
        for param in last_layer.parameters():
            param.requires_grad = True
    return model

def unfix_params(model):
    """ unfreeze model parameters """
    # activate layers 
    for param in model.parameters():
        param.requires_grad = True
    return model

class RandomRotate(object):
    """Implementation of random rotation.
    Randomly rotates an input image by a fixed angle. By default, we rotate
    the image by 90 degrees with a probability of 50%.
    This augmentation can be very useful for rotation invariant images such as
    in medical imaging or satellite imaginary.
    Attributes:
        prob:
            Probability with which image is rotated.
        angle:
            Angle by which the image is rotated. We recommend multiples of 90
            to prevent rasterization artifacts. If you pick numbers like
            90, 180, 270 the tensor will be rotated without introducing 
            any artifacts.
    
    """

    def __init__(self, prob: float = 0.5, angle: int = 90):
        self.prob = prob
        self.angle = angle

    def __call__(self, sample):
        """Rotates the images with a given probability.
        Args:
            sample:
                PIL image which will be rotated.
        
        Returns:
            Rotated image or original image.
        """
        prob = np.random.random_sample()
        if prob < self.prob:
            sample =  transforms.functional.rotate(sample, self.angle)
        return sample
        
def random_rotation_transform(
    rr_prob: float = 0.5,
    rr_degrees: Union[None, float, Tuple[float, float]] = 90,
    ) -> Union[RandomRotate, transforms.RandomApply]:
    if rr_degrees == 90:
        # Random rotation by 90 degrees.
        return RandomRotate(prob=rr_prob, angle=rr_degrees)
    else:
        # Random rotation with random angle defined by rr_degrees.
        return transforms.RandomApply([transforms.RandomRotation(degrees=rr_degrees)], p=rr_prob)

class Transforms_Segmentation:
    def __init__(self) -> None:
        pass

    def __call__(self, img: Image.Image) -> Image.Image:
        """extract silhouette"""
        x=np.array(img)
        x=x[:,:,1]*256+x[:,:,0] #0 for nothing, id (int) for tile
        x=np.array([x]*3, dtype=np.uint8).transpose((1,2,0)) # reshape (size,size)→(size,size,3)
        return Image.fromarray(x)

# logger
class logger_save():
    def __init__(self):
        self.tag=None
        self.level_dic = {
        'critical':logging.CRITICAL,
        'error':logging.ERROR,
        'warning':logging.WARNING,
        'info':logging.INFO,
        'debug':logging.DEBUG,
        'notset':logging.NOTSET
        }
        self.logger=None
        self.init_info=None
        self.level_console=None
        self.module_name=None

    def init_logger(self, module_name:str, outdir:str='', tag:str='',
                    level_console:str='warning', level_file:str='info'):
        #setting
        if len(tag)==0:
            tag = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.init_info={
            'level':self.level_dic[level_file],
            'filename':f'{outdir}/log_{tag}.txt',
            'format':'[%(asctime)s] [%(levelname)s] %(message)s',
            'datefmt':'%Y%m%d-%H%M%S'
            }
        self.level_console=level_console
        self.module_name=module_name
        #init
        logging.basicConfig(**self.init_info)
        logger = logging.getLogger(self.module_name)
        sh = logging.StreamHandler()
        sh.setLevel(self.level_dic[self.level_console])
        fmt = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            "%Y%m%d-%H%M%S"
            )
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        self.logger=logger

    def load_logger(self, filein:str=''):
        # load
        self.__dict__.update(pd.read_pickle(filein))
        #init
        logging.basicConfig(**self.init_info)
        logger = logging.getLogger(self.module_name)
        sh = logging.StreamHandler()
        sh.setLevel(self.level_dic[self.level_console])
        fmt = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            "%Y%m%d-%H%M%S"
            )
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        self.logger=logger

    def save_logger(self, fileout:str=''):
        pd.to_pickle(self.__dict__, fileout)

    def to_logger(self, name:str='', obj=None, skip_keys:set=set(), skip_hidden:bool=True):
        """ add instance information to logging """
        self.logger.info(name)
        for k,v in vars(obj).items():
            if k not in skip_keys:
                if skip_hidden:
                    if not k.startswith('_'):
                        self.logger.info('  {0}: {1}'.format(k,v))
                else:
                    self.logger.info('  {0}: {1}'.format(k,v))

def init_logger(
    module_name:str, outdir:str='', tag:str='',
    level_console:str='warning', level_file:str='info'
    ):
    """
    initialize logger
    
    """
    level_dic = {
        'critical':logging.CRITICAL,
        'error':logging.ERROR,
        'warning':logging.WARNING,
        'info':logging.INFO,
        'debug':logging.DEBUG,
        'notset':logging.NOTSET
        }
    if len(tag)==0:
        tag = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    logging.basicConfig(
        level=level_dic[level_file],
        filename=f'{outdir}/log_{tag}.txt',
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y%m%d-%H%M%S',
        )
    logger = logging.getLogger(module_name)
    sh = logging.StreamHandler()
    sh.setLevel(level_dic[level_console])
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        "%Y%m%d-%H%M%S"
        )
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger

def to_logger(
    logger, name:str='', obj=None, skip_keys:set=set(), skip_hidden:bool=True
    ):
    """ add instance information to logging """
    logger.info(name)
    for k,v in vars(obj).items():
        if k not in skip_keys:
            if skip_hidden:
                if not k.startswith('_'):
                    logger.info('  {0}: {1}'.format(k,v))
            else:
                logger.info('  {0}: {1}'.format(k,v))

# plot
def plot_progress(train_loss, val_loss, outdir:str="", name:str="progress", label:str="loss"):
    """ plot learning progress """
    fig, ax = plt.subplots()
    plt.rcParams['font.size'] = 18
    ax.plot(list(range(1, len(train_loss) + 1, 1)), train_loss, c='purple', label=f'train {label}')
    ax.plot(list(range(1, len(val_loss) + 1, 1)), val_loss, c='orange', label=f'val {label}')
    ax.set_xlabel('epoch')
    ax.set_ylabel(label)
    ax.grid()
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{outdir}/{name}.tif', dpi=100, bbox_inches='tight')

# learning tools
class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    add little changes from from https://github.com/Bjarten/early-stopping-pytorch/pytorchtools.py
    """
    def __init__(self, patience:int=7, delta:float=0, path:str='checkpoint.pt'):
        """
        Parameters
        ----------
            patience (int)
                How long to wait after last time validation loss improved.

            delta (float)
                Minimum change in the monitored quantity to qualify as an improvement.

            path (str): 
                Path for the checkpoint to be saved to.
   
        """
        self.patience = patience
        self.counter = 0
        self.best_score = np.Inf
        self.early_stop = False
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        if val_loss > self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), self.path)

    def delete_checkpoint(self):
        os.remove(self.path)

# Logger
class logger_save():
    def __init__(self):
        self.tag=None
        self.level_dic = {
        'critical':logging.CRITICAL,
        'error':logging.ERROR,
        'warning':logging.WARNING,
        'info':logging.INFO,
        'debug':logging.DEBUG,
        'notset':logging.NOTSET
        }
        self.logger=None
        self.init_info=None
        self.level_console=None
        self.module_name=None

    def init_logger(self, module_name:str, outdir:str='', tag:str='',
                    level_console:str='warning', level_file:str='info'):
        #setting
        if len(tag)==0:
            tag = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.init_info={
            'level':self.level_dic[level_file],
            'filename':f'{outdir}/log_{tag}.txt',
            'format':'[%(asctime)s] [%(levelname)s] %(message)s',
            'datefmt':'%Y%m%d-%H%M%S'
            }
        self.level_console=level_console
        self.module_name=module_name
        #init
        logging.basicConfig(**self.init_info)
        logger = logging.getLogger(self.module_name)
        sh = logging.StreamHandler()
        sh.setLevel(self.level_dic[self.level_console])
        fmt = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            "%Y%m%d-%H%M%S"
            )
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        self.logger=logger

    def load_logger(self, filein:str=''):
        # load
        self.__dict__.update(pd.read_pickle(filein))
        #init
        logging.basicConfig(**self.init_info)
        logger = logging.getLogger(self.module_name)
        sh = logging.StreamHandler()
        sh.setLevel(self.level_dic[self.level_console])
        fmt = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            "%Y%m%d-%H%M%S"
            )
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        self.logger=logger

    def save_logger(self, fileout:str=''):
        pd.to_pickle(self.__dict__, fileout)

    def to_logger(self, name:str='', obj=None, skip_keys:set=set(), skip_hidden:bool=True):
        """ add instance information to logging """
        self.logger.info(name)
        for k,v in vars(obj).items():
            if k not in skip_keys:
                if skip_hidden:
                    if not k.startswith('_'):
                        self.logger.info('  {0}: {1}'.format(k,v))
                else:
                    self.logger.info('  {0}: {1}'.format(k,v))

def init_logger(
    module_name:str, outdir:str='', tag:str='',
    level_console:str='warning', level_file:str='info'
    ):
    """
    initialize logger
    
    """
    level_dic = {
        'critical':logging.CRITICAL,
        'error':logging.ERROR,
        'warning':logging.WARNING,
        'info':logging.INFO,
        'debug':logging.DEBUG,
        'notset':logging.NOTSET
        }
    if len(tag)==0:
        tag = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    logging.basicConfig(
        level=level_dic[level_file],
        filename=f'{outdir}/log_{tag}.txt',
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y%m%d-%H%M%S',
        )
    logger = logging.getLogger(module_name)
    sh = logging.StreamHandler()
    sh.setLevel(level_dic[level_console])
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        "%Y%m%d-%H%M%S"
        )
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger