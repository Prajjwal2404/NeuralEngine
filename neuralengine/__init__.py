"""
NeuralEngine: A Framework for Building and Training Neural Networks

NeuralEngine provides core components for constructing, training and evaluating neural networks, with support for both CPU and GPU (CUDA) acceleration. Designed for extensibility, performance and ease of use, it is suitable for research, prototyping and production.
"""

from .config import Typed, DType, set_device, get_device
from .tensor import NoGrad, array
from .utils import *
from . import nn

__version__ = "0.6.0"
__all__ = ["Typed", "DType", "set_device", "get_device", "NoGrad", "array", "nn"] + utils.__all__