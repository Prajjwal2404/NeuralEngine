"""
NeuralEngine Neural Network Module (nn)

This module provides the core building blocks for constructing and training neural networks.
It includes layers, optimization algorithms, loss functions, evaluation metrics, data loading
utilities, and a high-level Model API.
"""

from .layers import Layer, Linear, LSTM, MultiplicativeAttention, MultiHeadAttention, \
Embedding, LayerNorm, Dropout, Flatten, Sigmoid, Tanh, ReLU, SiLU, Softmax
from .optim import Optimizer, SGD, Adam
from .loss import Loss, MSE, MAE, Huber, CrossEntropy, GaussianNLL, KLDivergence
from .metrics import Metric, RMSE, R2, ClassificationMetrics, Perplexity
from .model import Model
from .dataload import DataLoader

__all__ = [
    "Layer", "Linear", "LSTM", "MultiplicativeAttention", "MultiHeadAttention",
    "Embedding", "LayerNorm", "Dropout", "Flatten", "Sigmoid", "Tanh",
    "ReLU", "SiLU", "Softmax", "Optimizer", "SGD", "Adam", "Loss", "MSE", "MAE",
    "Huber", "CrossEntropy", "GaussianNLL", "KLDivergence", "Metric", "RMSE",
    "R2", "ClassificationMetrics", "Perplexity", "Model", "DataLoader"
]