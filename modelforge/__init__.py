"""ModelForge — machine learning from scratch, from data to deployed model.

Built for the course "Fundamentals of Machine Learning: From Data to Deployed
Models". Every algorithm is implemented by hand in pure Python, with no numpy,
no scikit-learn and no GPU:

  linalg           vectors, matrices, softmax, a linear solver          Module 2
  datasets         blobs / moons / circles / rare-event generators       Module 1
  metrics          accuracy, confusion matrix, precision/recall/F1       Module 3
  regression       linear regression on MSE (batch / mini-batch / SGD)   Module 2
  linear           softmax regression trained by gradient descent        Modules 2-3
  neighbors        k-nearest neighbours                                  Module 3
  tree             CART decision tree (Gini or entropy), random forest   Module 3
  paradigms        k-means (unsupervised) and Q-learning (reinforcement) Module 1
  neural           multilayer perceptron, 6 activations, backprop,
                   gradient check, early stopping, fine-tuning           Module 4
  architectures    convolution, RNN/BPTT, clipping, LSTM, GRU,
                   self-attention, positional encoding                   Module 5
  registry         versioned registry, quality gates, rollback, PSI, KS  Module 6
  mlops            data validation, feature store, canary, shadow,
                   monitoring decisions, out-of-distribution guard       Module 6
  frontier         adversarial examples, overconfidence, importance      Module 6
  experiments, deep_experiments, ops_experiments, course_modules
                   one measured experiment per lesson                    Modules 1-6
  viz / report     SVG plots and a self-contained HTML lab report
"""

__version__ = "1.1.0"

from .datasets import make_blobs, make_moons, make_circles, train_test_split, StandardScaler
from .linear import SoftmaxRegression
from .neighbors import KNearestNeighbors
from .tree import DecisionTree, RandomForest
from .neural import MLP
from .metrics import accuracy, evaluate, confusion_matrix, precision_recall_f1, cross_entropy
from .pipeline import compare, depth_sweep, knn_sweep, prepare, default_models
from .registry import ModelRegistry, drift_report, psi, shift

__all__ = [
    "make_blobs", "make_moons", "make_circles", "train_test_split", "StandardScaler",
    "SoftmaxRegression", "KNearestNeighbors", "DecisionTree", "RandomForest", "MLP",
    "accuracy", "evaluate", "confusion_matrix", "precision_recall_f1", "cross_entropy",
    "compare", "depth_sweep", "knn_sweep", "prepare", "default_models",
    "ModelRegistry", "drift_report", "psi", "shift",
]

