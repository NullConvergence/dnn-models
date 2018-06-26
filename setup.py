from setuptools import find_packages
from setuptools import setup

setup(name='dnn-models',
      version='1.0.0',
      url='https://github.com/NullConvergence/dnn-models',
      license='MIT',
      install_requires=[
          'nose',
          'pycodestyle',
          'scipy',
          'matplotlib',
          "mnist ~= 0.2",
          "numpy",
          'cleverhans',
          'tensorflow'
      ],
      packages=find_packages())
