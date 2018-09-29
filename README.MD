
The library was created in order to extend [Cleverhans](https://github.com/tensorflow/cleverhans) [models](https://github.com/tensorflow/cleverhans/blob/master/cleverhans/model.py).
The main goal is to ease the design process for new models by introducing the ``` train_graph.py ``` abstraction and other layers such as ``` lrn.py ```.

The inspiration comes from [Cleverhans](https://github.com/tensorflow/cleverhans/blob/master/cleverhans_tutorials/tutorial_models.py). For the moment all models are extended to use [name_scopes](https://www.tensorflow.org/api_docs/python/tf/name_scope). Moreover, new architectures will be implemented in the [``` get_model.py ```](https://github.com/NullConvergence/dnn-models/blob/master/get_model.py) file.

### Installation

#### ``` pip ``` Installation: 

``` pip install dnnmodels ```

#### Manual Installation:

``` git clone https://github.com/NullConvergence/dnnmodels ```

``` pip install -e ./dnnmodels ```


### Examples
* At the moment there is only one example: [``` mnist.py ```](https://github.com/NullConvergence/dnn-models/blob/master/examples/mnist_train.py)
