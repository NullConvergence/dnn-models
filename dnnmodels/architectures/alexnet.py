from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

from dnnmodels.primitives.layers import *

class Alexnet(object):
    def __init__():
        """
        Model for AlexNet ImageNet
        - Alex Krizhevsky, Ilya Sutskever & Geoffrey E. Hinton.
        ImageNet Classification with Deep Convolutional Neural Networks. NIPS, 2012.
        Link:
        - https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks
        """
        layers = [
                Conv2D(kernel_shape=(11, 11), strides=(4, 4), padding='SAME', nr_filters=nb_filters, name='conv_1'),
                ReLU(name='relu_1'),
                LocalResponseNormalization(name='lrn_1'),
                MaxPool(ksize=(3, 3), strides=(2, 2),
                        padding='VALID', name='max_pool_1'),

                Conv2D(kernel_shape=(5, 5), strides=(1, 1), padding='SAME', nr_filters=192, name='conv_2'),
                ReLU(name='relu_2'),
                LocalResponseNormalization(name='lrn_2'),
                MaxPool(ksize=(3, 3), strides=(2, 2),
                        padding='VALID', name='max_pool_2'),

                Conv2D(kernel_shape=(3, 3), strides=(1, 1), padding='SAME', nr_filters=384, name='conv_3'),
                ReLU(name='relu_3'),

                Conv2D(kernel_shape=(3, 3), strides=(1, 1), padding='SAME', nr_filters=256, name='conv_4'),
                ReLU(name='relu_4'),

                Conv2D(kernel_shape=(3, 3), strides=(1, 1), padding='SAME', nr_filters=256, name='conv_5'),
                ReLU(name='relu_5'),
                MaxPool(ksize=(3, 3), strides=(2, 2),
                        padding='VALID', name='max_pool_5'),

                Flatten(name='flatten_1'),
                FullyConnected(nr_hidden=4096, name='fc_1'),
                Dropout(name='drop_1'),

                FullyConnected(nr_hidden=4096, name='fc_2'),
                Dropout(name='drop_2'),

                FullyConnected(nr_hidden=nb_classes, name='logits'),

                Softmax(name='probs')
            ]

        model = MLP(layers, input_shape)
        return model