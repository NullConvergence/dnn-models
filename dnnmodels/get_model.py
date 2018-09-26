from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

from .primitives.layers import *
from .architectures import *

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

def alex_net(nb_filters=64, nb_classes=1000, input_shape=(None, None, None, None)):
        """
            Instantiate Alexnet model and return it
        """
        model = Alexnet(nb_filters=nb_filters, nb_classes=nb_classes, input_shape=input_shape)
        return model
        
