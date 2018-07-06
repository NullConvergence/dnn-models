from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
from cleverhans.model import Model
from .softmax import Softmax


class MLP(Model):
    """
    An example of a bare bones multilayer perceptron (MLP) class.
    : taken from https://github.com/tensorflow/cleverhans/blob/master/cleverhans_tutorials/tutorial_models.py
    """

    def __init__(self, layers, input_shape):
        """
        Initializes layers & layer names
        """
        super(MLP, self).__init__()

        self.layer_names = []
        self.layers = layers
        self.input_shape = input_shape
        if isinstance(layers[-1], Softmax):
            layers[-1].name = 'probs'
            layers[-2].name = 'logits'
        else:
            layers[-1].name = 'logits'
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'name'):
                name = layer.name
            else:
                name = layer.__class__.__name__ + str(i)
                layer.name = name
            self.layer_names.append(name)

            layer.set_input_shape(input_shape)
            input_shape = layer.get_output_shape()

    def fprop(self, x, set_ref=False):
        """
        Implements inference as forward propagation for each layer
        """
        states = []
        for layer in self.layers:
            if set_ref:
                layer.ref = x
            x = layer.fprop(x)
            assert x is not None
            states.append(x)
        states = dict(zip(self.get_layer_names(), states))
        return states

            def reng(self, ]x):
        """
        Reverse engineers all layers (if possible) and returns a dic
        TODO: Think about moving this to Model in a cleverhans fork
        """
        try:
            states = []
            for layer in self.layers:
                x = layer.reng(x)
                states.append(x)
            states = dict(zip(self.get_layer_names(), states))
            return states
        except Exception as e:
            # TODO: log this
            raise e

    def get_params(self):
        """
        Returns all parameters from all layers
        """
        out = []
        for layer in self.layers:
            for param in layer.get_params():
                if param not in out:
                    out.append(param)
        return out
