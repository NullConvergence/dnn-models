Some base [Tensorflow](http://tensorflow.org) models used to build DNNs.
They are addressed to extend [Cleverhans](https://github.com/tensorflow/cleverhans) [models](https://github.com/tensorflow/cleverhans/blob/master/cleverhans/model.py) and to be used for adversarial attacks.
However, one can use the models for other tasks too.

The inspiration comes from [Cleverhans](https://github.com/tensorflow/cleverhans/blob/master/cleverhans_tutorials/tutorial_models.py), but all models are extended to use [name_scopes](https://www.tensorflow.org/api_docs/python/tf/name_scope). Moreover, more architectures will be implemented in the [``` get_model.py ```](https://github.com/NullConvergence/dnn-models/blob/master/get_model.py) file.

