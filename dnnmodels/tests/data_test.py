
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest


import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

from data import *

class TestData(unittest.TestCase):
    def test_imagenet(self):
        img_net = ImageNet('/Users/alexandru/Documents/Projects/Tensorflow/dnnmodels/downloaded_data/imagenet')
        pass


if __name__ == '__main__':
    unittest.main()