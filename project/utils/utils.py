# %%
import os
import torch

import random
import shutil

import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
# %%
def del_folder(path, *args):
    '''
    delete the folder which path/version

    Args:
        path (str): path
        version (str): version
    '''
    if os.path.exists(os.path.join(path, *args)):
        shutil.rmtree(os.path.join(path, *args))


def make_folder(path, *args):
    '''
    make folder which path/version

    Args:
        path (str): path
        version (str): version
    '''
    if not os.path.exists(os.path.join(path, *args)):
        os.makedirs(os.path.join(path, *args))
        print("success make dir! where: %s " % os.path.join(path, *args))
    else:
        print("The target path already exists! where: %s " % os.path.join(path, *args))


def tensor2var(x, grad=False):
    '''
    put tensor to gpu, and set grad to false

    Args:
        x (tensor): input tensor
        grad (bool, optional):  Defaults to False.

    Returns:
        tensor: tensor in gpu and set grad to false 
    '''
    if torch.cuda.is_available():
        x = x.cuda()
        x.requires_grad_(grad)
    return x


def var2tensor(x):
    '''
    put date to cpu

    Args:
        x (tensor): input tensor 

    Returns:
        tensor: put data to cpu
    '''
    return x.data.cpu()


def var2numpy(x):
    return x.data.cpu().numpy()

def get_ckpt_path(config):
    '''
    get the finall train ckpt, for the pretrain model, or some task.

    Args:
        config (parser): parameters of the training

    Returns:
        string: final ckpt file path
    '''
    
    ckpt_path = os.path.join('/workspace/Walk_Video_PyTorch', 'logs', config.model, config.version, 'checkpoints')
    file_name = os.listdir(ckpt_path)[0]

    ckpt_file_path = os.path.join(ckpt_path, file_name)

    return ckpt_file_path

def plot(imgs, **imshow_kwargs):
    '''
    receive a tensor, and plot the figure

    Args:
        imgs (tensor): a batch of tensor
    '''
        
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            img = F.to_pil_image(img.to("cpu"))
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()