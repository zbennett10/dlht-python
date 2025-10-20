"""
DLHT - Distributed Learned Hash Table

A high-performance DHT with machine learning-based routing using Recursive Model Index (RMI).

Features:
- Learned routing with adaptive training
- Chord-style distributed architecture
- Range query support
- Federated model averaging
- Error-bounded predictions

Usage:
    from dlht import LEADNode, LEADConfig, AdaptiveTrainingManager

    # Simple single-node usage
    config = LEADConfig(ip='localhost', base_port=9000)
    node = LEADNode(config)
    node.start()

    node.put(key=12345, value={"data": "value"})
    value = node.get(key=12345)
    results = node.range_query(hash_min=10000, hash_max=20000)

    # Adaptive training
    node.retrain_model(force=True)
    metrics = node.training_manager.get_metrics()
"""

__version__ = "0.1.0"
__author__ = "COTI Network"
__license__ = "MIT"

# Core DHT components
from .node import LEADNode
from .peer import LEADPeer, FingerEntry
from .config import LEADConfig

# Learned index models
from .models import (
    RecursiveModelIndex,
    LinearModel,
    CubicModel
)

# Adaptive training
from .adaptive_training import (
    AdaptiveTrainingManager,
    TrainingConfig,
    TrainingMetrics,
    federated_average_models
)

# Utilities and exceptions
from .exceptions import (
    NetworkException,
    RPCException,
    NodeNotReadyException
)

from .utils import (
    RPCMessage,
    peer_hash,
    distance
)

__all__ = [
    # Main classes
    "LEADNode",
    "LEADPeer",
    "LEADConfig",
    "FingerEntry",

    # Models
    "RecursiveModelIndex",
    "LinearModel",
    "CubicModel",

    # Adaptive training
    "AdaptiveTrainingManager",
    "TrainingConfig",
    "TrainingMetrics",
    "federated_average_models",

    # Exceptions
    "NetworkException",
    "RPCException",
    "NodeNotReadyException",

    # Utils
    "RPCMessage",
    "peer_hash",
    "distance",

    # Metadata
    "__version__",
    "__author__",
    "__license__",
]