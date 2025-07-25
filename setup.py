from setuptools import find_packages
from distutils.core import setup

setup(name='unitree_rl_gym',
      version='1.0.0',
      author='Unitree Robotics',
      license="BSD-3-Clause",
      packages=find_packages(),
      author_email='support@unitree.com',
      description='Template RL environments for Unitree Robots',
      install_requires=['rsl-rl', 'matplotlib','tensorboard', 'pyyaml'])
