from setuptools import setup
from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='dnnmodels',
      version='1.0.0',
      url='https://github.com/NullConvergence/dnnmodels',
      license='MIT',
      author='NullConvergence',
      description="Simple models inspired by Cleverhans that help create DNN Architectures",
      install_requires=[
          'cleverhans',
          'tensorflow'
      ],
      packages=find_packages())
