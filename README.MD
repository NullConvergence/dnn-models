The library was created in order to extend [Cleverhans](https://github.com/tensorflow/cleverhans) [models](https://github.com/tensorflow/cleverhans/blob/master/cleverhans/model.py).
The main goal is to ease the design process for new models by introducing the ``` train_graph.py ``` abstraction and other layers such as ``` lrn.py ```.

The inspiration comes from [Cleverhans](https://github.com/tensorflow/cleverhans/blob/master/cleverhans_tutorials/tutorial_models.py). For the moment all models are extended to use [name_scopes](https://www.tensorflow.org/api_docs/python/tf/name_scope). Moreover, new architectures will be implemented in the [``` get_model.py ```](https://github.com/NullConvergence/dnn-models/blob/master/get_model.py) file.

### Usage

* Clone project: ``` git clone https://github.com/NullConvergence/dnn-models ```
* Import prefered dnn from *(``` get_model.py ```)[https://github.com/NullConvergence/dnn-models/blob/master/lib/get_model.py]: e.g.  ``` from lib.get_models import basic_cnn ```
* Import trainer from [``` train_graph.py ```](https://github.com/NullConvergence/dnn-models/blob/master/lib/train_graph.py): e.g.  ``` from lib.train_graph import Trainer ```
* Instantiate Model: e.g. ``` my_model = basic_cnn() ``` 
* Instantiate Trainer: e.g. ``` trainer = Trainer() ``` 


### Examples
* At the moment there is only one example: [``` mnist.py ```](https://github.com/NullConvergence/dnn-models/blob/master/examples/mnist_train.py)