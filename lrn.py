from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
from models.layer import Layer


class LocalResponseNormalization(Layer):

    def __init__(self, alpha, beta, depth_radius, bias, name='lrn'):
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

    def set_output_shape(self, output_shape):
        """
        Just sets the output shape
        """
        self.output_shape = output_shape

    def fprop(self, input):
        """
        Implements forward propagation as required by CleverHans Model Interface
        """
        with tf.name_scope(self.name):
            return tf.nn.local_response_normalization(input,
                                                      alpha=self.alpha,
                                                      beta=self.beta,
                                                      depth_radius=self.depth_radius,
                                                      bias=self.bias)

    def get_params(self):
        """
        Implements get_params as required by CleverHans Model Interface
        """
        return [self.alpha, self.beta, self.depth_radius, self.bias]
