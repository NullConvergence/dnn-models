from abc import ABCMeta


class Layer(object):

    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    def fprop(self):
        raise NotImplementedError('Forward Propagation Not Implemented :o\t')

    def reng(self):
        raise NotImplementedError('Reverse Engineering not implemented :o\t')

    def set_input_shape(self, shape):
        try:
            self.input_shape = shape
            self.set_output_shape(shape)
        except Exception as e:
            raise e

    def set_output_shape(self, shape):
        try:
            self.output_shape = shape
        except Exception as e:
            raise e
