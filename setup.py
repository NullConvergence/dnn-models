from setuptools import find_packages
from setuptools import setup

setup(name='dnn_models',
      version='1.0.0',
      url='https://github.com/NullConvergence/dnn-models',
      license='MIT',
      author='NullConvergence',
      install_requires=[
          'cleverhans',
          'tensorflow'
      ],
      packages=find_packages())
