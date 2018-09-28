from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
from .layer import Layer


class ReLU(Layer):

    def __init__(self, name='relu'):
        """
        All parameters are processed by __dict__.update(locals())
        Unfortunately the method also processes 'self' so it is removed
        """
        self.__dict__.update(locals())
        del self.self

    def fprop(self, input):
        """
        Implements forward propagation as required by CleverHans Model Interface
        """
        with tf.name_scope(self.name):
            return tf.nn.relu(input, name='relu')

    def reng(self, input):
        """
        Implements reverse engineering of layer - in this case relu
        """
        with tf.name_scope(self.name):
            return tf.nn.relu(input, name='de-relu')

    def get_params(self):
        """
        Implements get_params as required by CleverHans Model Interface
        """
        return []
