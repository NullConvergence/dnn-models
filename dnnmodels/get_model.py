from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

# TODO: find a better way for this
from .conv2d import Conv2D
from .dropout import Dropout
from .flatten import Flatten
from .fully_connected import FullyConnected
from .layer import Layer
from .lrn import LocalResponseNormalization
from .max_pool import MaxPool
from .relu import ReLU
from .softmax import Softmax
from .mlp import MLP


def basic_cnn(nb_filters=64, nb_classes=10, input_shape=(None, 28, 28, 1)):
    """
    The default input shape is set to MNIST images
    """
    layers = [Conv2D((8, 8), (2, 2), "SAME", nb_filters, name='conv_1'),
              ReLU(name='relu_1'),
              Conv2D((6, 6), (2, 2), "VALID",
                     nb_filters*2, name='conv_2'),
              ReLU(name='relu_2'),
              Conv2D((5, 5), (1, 1), "VALID",
                     nb_filters*2, name='conv_3'),
              ReLU(name='relu_3'),
              Flatten(name='fc_1'),
              FullyConnected(nb_classes, name='logits'),
              Softmax(name='probs')]

    model = MLP(layers, input_shape)

    return model


def alexnet(nb_filters=64, nb_classes=1000, input_shape=[None, 227, 227, 3]):
    """
    Model for AlexNet ImageNet
    - Alex Krizhevsky, Ilya Sutskever & Geoffrey E. Hinton.
    ImageNet Classification with Deep Convolutional Neural Networks. NIPS, 2012.
    Link:
    - https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks
    """
    layers = [
        Conv2D(kernel_shape=(11, 11), strides=(4, 4), 'SAME', nr_filters=nb_filters, name='conv_1'),
        ReLU(name='relu_1'),
        LocalResponseNormalization(name='lrn_1'),
        MaxPool(ksize=(3, 3), strides=(2, 2),
                padding='VALID', name='max_pool_1'),

        Conv2D(kernel_shape=(5, 5), strides=(1, 1), 'SAME', nr_filters=192, name='conv_2')
        ReLU(name='relu_2')
        LocalResponseNormalization(name='lrn_2')
        MaxPool(ksize=(3, 3), strides=(2, 2),
                padding='VALID', name='max_pool_2'),

        Conv2D(kernel_shape=(3, 3), strides=(1, 1), 'SAME', nr_filters=384, name='conv_3')
        ReLU(name='relu_3')

        Conv2D(kernel_shape=(3, 3), strides=(1, 1), 'SAME', nr_filters=256, name='conv_4')
        ReLU(name='relu_4')

        Conv2D(kernel_shape=(3, 3), strides=(1, 1), 'SAME', nr_filters=256, name='conv_5')
        ReLU(name='relu_5')
        MaxPool(ksize=(3, 3), strides=(2, 2),
                padding='VALID', name='max_pool_5'),

        Flatten(name='flatten_1')
        FullyConnected(nr_hidden=4096, name='fc_1')
        Dropout(name='drop_1')

        FullyConnected(nr_hidden=4096, name='fc_2')
        Dropout(name='drop_2')

        FullyConnected(nr_hidden=nb_classes, name='logits')

        Softmax(name='probs')

        model = MLP(layers, input_shape)
        return model
    ]
