from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

from .data import Data
from .constants import IMAGENET_URL

class ImageNet(Data):
    def __init__(self, folder):
        if(folder is None):
            raise('Please provide folder')
        else:
            super().__init__(folder, IMAGENET_URL)
