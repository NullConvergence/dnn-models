import tensorflow as tf
from cleverhans.utils_mnist import data_mnist

# This could be changed to from dnnmodels.etc import x -> when the dnnmodels
# package is installed
import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from dnnmodels.get_model import basic_cnn
from dnnmodels.primitives.train_graph import Trainer

# create session
sess = tf.Session()

# Get MNIST data data
train_start = 0
train_end = 60000
test_start = 0
test_end = 10000

X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
                                              train_end=train_end,
                                              test_start=test_start,
                                              test_end=test_end)

# Use label smoothing
assert Y_train.shape[1] == 10
label_smooth = .1
Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

# Configure train params
train_params = {
    'nb_epochs': 1,
    'batch_size': 128,
    'learning_rate': 0.001,
    'save_model': False
}

# Configure data params
data_params = {
    'x_shape': [None, 28, 28, 1],
    'y_shape': [None, 10],
    'X_train': X_train,
    'Y_train': Y_train,
    'X_test': X_test,
    'Y_test': Y_test
}


def main():
    # create model
    model = basic_cnn()
    # create trainer
    trainer = Trainer(sess, model, data_params, train_params)
    trainer.train()


if __name__ == '__main__':
    # TODO: parse
    main()
