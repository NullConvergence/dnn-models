import tensorflow as tf

from dnnmodels.get_model import alexnet
from dnnmodels.train_graph import Trainer

# create session
sess = tf.Session()


def get_data():
    pass


X_train, Y_train, X_test, Y_test = ()

# Configure train params
train_params = {
    'nb_epochs': 1,
    'batch_size': 128,
    'learning_rate': 0.001,
    'save_model': False
}

# Configure data params
data_params = {
    'x_shape': [None, 227, 227, 3],
    'y_shape': [None, 1000],
    'X_train': X_train,
    'Y_train': Y_train,
    'X_test': X_test,
    'Y_test': Y_test
}


def main():
    # create model
    model = alexnet()
    # create trainer
    trainer = Trainer(sess, model, data_params, train_params)
    trainer.train()


if __name__ == '__main__':
    # TODO: parse args
    main()
