from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
from .layer import Layer


class Dropout(Layer):

    def __init__(self, name='dropout', keep_prob=0.5):
        """
        All parameters are processed by __dict__.update(locals())
        Unfortunately the method also processes 'self' so it is removed
        """
        self.__dict__.update(locals())
        del self.self

    def set_input_shape(self, input_shape):
        """
        This method initializes all layer variables
        """
        self.input_shape = input_shape
        self.set_output_shape(input_shape)

    def set_output_shape(self, shape):
        """
        Just sets the output shape
        """
        self.output_shape = shape

    def fprop(self, input):
        """
        Implements forward propagation as required by CleverHans Model Interface
        """
        with tf.name_scope(self.name):
            return tf.nn.dropout(input, keep_prob=self.keep_prob)

    def get_params(self):
        """
        Implements get_params as required by CleverHans Model Interface
        """
        return []
