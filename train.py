import argparse
import os, sys
import shutil
import time
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib
import matplotlib.pyplot as plt
import random
import numpy as np
import statistics
import csv
from resnet import *

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--dataset', default='PCam', type=str,
                    help='dataset (PCam [default])')
parser.add_argument('--model', default='res32', type=str,
                    help='model (res32 [default])')
args = parser.parse_args()