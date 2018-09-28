from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
from .layer import Layer


class MaxPool(Layer):

    def __init__(self, ksize, strides, padding, name='maxpool', argmax=False):
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
        shape = list(shape)
        # set batch size to 1
        shape[0] = 1
        # get dummy output
        dummy_batch = tf.zeros(shape)
        dummy_output = self.fprop(dummy_batch)
        # get shape of dummy output
        output_shape = [int(e) for e in dummy_output.get_shape()]
        output_shape[0] = 1
        # set output_shape
        self.output_shape = tuple(output_shape)

    def fprop(self, input):
        """
        Implements forward propagation 
        """
        with tf.name_scope(self.name):
            # check if we need maxpool with argmax - which
            # also returns maxpool indices
            if not self.argmax:
                return tf.nn.max_pool(input,
                                      ksize=(1,) + tuple(self.ksize) + (1,),
                                      strides=(1,) +
                                      tuple(self.strides) + (1,),
                                      padding=self.padding,
                                      name='max_pool')
            else:
                # TODO: Save indices somewhere
                return tf.nn.max_pool_with_argmax(input,
                                                  ksize=(
                                                      1,) + tuple(self.ksize) + (1,),
                                                  strides=(1,) +
                                                  tuple(self.strides) + (1,),
                                                  Targmax=tf.int64,
                                                  padding=self.padding,
                                                  name='max_pool'
                                                  )

    def reng(self, input):
        """
        Implements reverse engineering of layer - unpooling
        """
        _name = 'de-' + self.name
        if not self.argmax:
            return self._unpool2d(input)
        else:
            return self._unpool2d_argmax()

    def get_params(self):
        """
        Implements get_params 
        """
        return [self.ksize, self.strides, self.padding]

    def _unpool2d(self, input):
        """
        Unpool implementation with reshape & concat
        https://github.com/tensorflow/tensorflow/issues/2169
        """
        with tf.name_scope(self.name) as scope:
            sh = input.get_shape().as_list()
            dim = len(sh[1:-1])
            out = (tf.reshape(input, [-1] + sh[-dim:]))
            for i in range(dim, 0, -1):
                out = tf.concat([out, tf.zeros_like(out)], i)
            out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
            out = tf.reshape(out, out_size, name='de-pool')
        return out

    def _unpool2d_argmax(self, pool, ind):
        """

        """
        stride = (1,) + tuple(self.strides) + (1,)
        with tf.variable_scope(self.name):
            ind_shape = tf.shape(ind)

            input_shape = tf.shape(pool)
            output_shape = [input_shape[0],
                            input_shape[1] * stride[1],
                            input_shape[2] * stride[2],
                            input_shape[3]]

            flat_input_size = tf.reduce_prod(input_shape)
            flat_output_shape = [output_shape[0],
                                 output_shape[1] * output_shape[2] * output_shape[3]]

            pool_ = tf.reshape(pool, [flat_input_size])
            batch_range = tf.reshape(
                tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype),
                shape=[input_shape[0], 1, 1, 1])
            b = tf.ones_like(ind) * batch_range
            b1 = tf.reshape(b, [flat_input_size, 1])
            ind_ = tf.reshape(ind, [flat_input_size, 1])
            ind_ = tf.concat([b1, ind_], 1)

            ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(
                flat_output_shape, tf.int64))
            ret = tf.reshape(ret, output_shape, name='de-pool')

            set_input_shape = pool.get_shape()
            set_output_shape = [set_input_shape[0],
                                set_input_shape[1] * stride[1],
                                set_input_shape[2] * stride[2],
                                set_input_shape[3]]
            ret.set_shape(set_output_shape)
            return ret
