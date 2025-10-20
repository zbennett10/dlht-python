"""Federated Recursive Model (FRM) for distributed DLHT training.

This module implements the Federated Recursive Model as described in the LEAD
paper (arXiv:2508.14239, Section III-E3). FRM enables decentralized model
training where each peer refines its segment of leaf models based on locally
observed data changes, without requiring a central coordinator.

Key features:
  - Distributed leaf model updates per peer
  - Parameter aggregation via transient coordinators
  - Version-based model synchronization
  - Heartbeat-based readiness coordination
  - Minimal parameter transfer (only changed segments)

Typical usage example:

  frm = FederatedRecursiveModel(
      num_leaf_models=100,
      model_type='linear',
      update_threshold=0.4
  )

  # Peer updates local leaf model
  frm.update_leaf_model(leaf_index=42, keys=new_keys, positions=new_positions)

  # Check if ready for federated update
  if frm.should_trigger_federated_update():
      # Become transient coordinator
      params = frm.get_leaf_parameters()
      # Aggregate from neighbors...
      frm.apply_aggregated_parameters(aggregated_params)
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Deque

import numpy as np
from numpy.typing import NDArray

# Google Style: Use absolute imports for clarity
from dlht.models import LinearModel, CubicModel, RecursiveModelIndex
from dlht.exceptions import ModelException


logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Supported leaf model types for FRM.

    Attributes:
        LINEAR: Linear regression model (smallest size, ~1MB)
        CUBIC: Cubic polynomial model (larger size, ~12MB, better accuracy)
    """
    LINEAR = 'linear'
    CUBIC = 'cubic'


@dataclass
class LeafModelParameters:
    """Parameters for a single leaf model in the RMI.

    This class encapsulates the trainable parameters of a leaf model along
    with metadata for tracking updates and versioning.

    Attributes:
        leaf_index: Index of this leaf model in the RMI structure
        coefficients: Model coefficients (e.g., [slope, intercept] for linear)
        min_key: Minimum key this leaf model handles
        max_key: Maximum key this leaf model handles
        num_samples: Number of samples used to train this leaf
        last_updated: Timestamp of last update
        version: Version number of this leaf model
    """
    leaf_index: int
    coefficients: NDArray[np.float64]
    min_key: float
    max_key: float
    num_samples: int
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = 0

    def to_bytes(self) -> bytes:
        """Serialize leaf parameters for network transfer.

        Returns:
            Serialized byte representation (~12-48 bytes depending on model type)
        """
        # Format: leaf_index(4) + version(4) + num_coeffs(4) + coeffs(n*8) + bounds(2*8)
        import struct

        data = struct.pack('<III', self.leaf_index, self.version, len(self.coefficients))
        data += self.coefficients.tobytes()
        data += struct.pack('<dd', self.min_key, self.max_key)
        return data

    @classmethod
    def from_bytes(cls, data: bytes) -> LeafModelParameters:
        """Deserialize leaf parameters from network transfer.

        Args:
            data: Serialized byte representation

        Returns:
            Reconstructed LeafModelParameters instance

        Raises:
            ValueError: If data is malformed
        """
        import struct

        if len(data) < 12:
            raise ValueError(f"Invalid data length: {len(data)} < 12")

        leaf_index, version, num_coeffs = struct.unpack('<III', data[:12])
        offset = 12

        coeffs_bytes = num_coeffs * 8
        coefficients = np.frombuffer(data[offset:offset + coeffs_bytes], dtype=np.float64)
        offset += coeffs_bytes

        min_key, max_key = struct.unpack('<dd', data[offset:offset + 16])

        return cls(
            leaf_index=leaf_index,
            coefficients=coefficients,
            min_key=min_key,
            max_key=max_key,
            num_samples=0,  # Not transferred
            version=version
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation suitable for RPC transmission
        """
        return {
            'leaf_index': int(self.leaf_index),
            'coefficients': self.coefficients.tolist(),
            'min_key': float(self.min_key),
            'max_key': float(self.max_key),
            'num_samples': int(self.num_samples),
            'last_updated': self.last_updated.isoformat(),
            'version': int(self.version)
        }

    @classmethod
    def from_dict(cls, data: dict) -> LeafModelParameters:
        """Create instance from dictionary.

        Args:
            data: Dictionary representation from RPC

        Returns:
            LeafModelParameters instance
        """
        return cls(
            leaf_index=data['leaf_index'],
            coefficients=np.array(data['coefficients'], dtype=np.float64),
            min_key=data['min_key'],
            max_key=data['max_key'],
            num_samples=data['num_samples'],
            last_updated=datetime.fromisoformat(data['last_updated']),
            version=data['version']
        )

    def __repr__(self) -> str:
        return (f"LeafModelParameters(leaf_index={self.leaf_index}, "
                f"version={self.version}, "
                f"coeffs={self.coefficients.tolist()}, "
                f"range=[{self.min_key:.2f}, {self.max_key:.2f}])")


@dataclass
class FRMMetrics:
    """Performance metrics for Federated Recursive Model.

    Tracks key performance indicators for FRM operations including:
    - Update latency and throughput
    - Parameter transfer overhead
    - Convergence metrics
    - Network efficiency

    Attributes:
        total_updates: Total number of federated updates performed
        total_leaf_updates: Total number of individual leaf model updates
        avg_update_latency_ms: Average time to complete federated update
        last_update_time: Timestamp of last federated update
        coordinator_count: Number of times this node became coordinator
        parameters_sent_bytes: Total bytes of parameters sent
        parameters_received_bytes: Total bytes of parameters received
        neighbor_participation_rate: Average % of neighbors participating
        model_convergence_score: Score indicating model stability (0-1)
    """
    total_updates: int = 0
    total_leaf_updates: int = 0
    avg_update_latency_ms: float = 0.0
    last_update_time: Optional[datetime] = None
    coordinator_count: int = 0
    parameters_sent_bytes: int = 0
    parameters_received_bytes: int = 0
    neighbor_participation_rate: float = 0.0
    model_convergence_score: float = 0.0

    # Internal tracking (not exposed in to_dict)
    _update_latencies: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    _participation_rates: Deque[float] = field(default_factory=lambda: deque(maxlen=100))

    def record_update(
        self,
        latency_ms: float,
        coordinator: bool,
        peers_participated: int,
        total_peers: int,
        bytes_sent: int,
        bytes_received: int
    ) -> None:
        """Record a federated update completion.

        Args:
            latency_ms: Time taken for update in milliseconds
            coordinator: Whether this node was coordinator
            peers_participated: Number of peers that participated
            total_peers: Total number of peers in network
            bytes_sent: Bytes of parameters sent
            bytes_received: Bytes of parameters received
        """
        self.total_updates += 1
        if coordinator:
            self.coordinator_count += 1

        self._update_latencies.append(latency_ms)
        self.avg_update_latency_ms = float(np.mean(list(self._update_latencies)))

        participation_rate = peers_participated / total_peers if total_peers > 0 else 0.0
        self._participation_rates.append(participation_rate)
        self.neighbor_participation_rate = float(np.mean(list(self._participation_rates)))

        self.parameters_sent_bytes += bytes_sent
        self.parameters_received_bytes += bytes_received
        self.last_update_time = datetime.now(timezone.utc)

    def record_leaf_update(self) -> None:
        """Record a local leaf model update."""
        self.total_leaf_updates += 1

    def update_convergence_score(self, score: float) -> None:
        """Update model convergence score (0-1).

        Args:
            score: Convergence score, higher is better (0-1)
        """
        self.model_convergence_score = max(0.0, min(1.0, score))

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for reporting.

        Returns:
            Dict with all public metrics
        """
        return {
            'total_updates': self.total_updates,
            'total_leaf_updates': self.total_leaf_updates,
            'avg_update_latency_ms': round(self.avg_update_latency_ms, 2),
            'last_update_time': self.last_update_time.isoformat() if self.last_update_time else None,
            'coordinator_count': self.coordinator_count,
            'parameters_sent_bytes': self.parameters_sent_bytes,
            'parameters_received_bytes': self.parameters_received_bytes,
            'neighbor_participation_rate': round(self.neighbor_participation_rate, 3),
            'model_convergence_score': round(self.model_convergence_score, 3),
            'throughput_updates_per_hour': self._calculate_throughput()
        }

    def _calculate_throughput(self) -> float:
        """Calculate updates per hour based on recent history."""
        if not self.last_update_time or self.total_updates == 0:
            return 0.0

        # Simple approximation: if we have latency data, estimate throughput
        if self._update_latencies:
            avg_latency_hours = self.avg_update_latency_ms / (1000 * 3600)
            return 1.0 / avg_latency_hours if avg_latency_hours > 0 else 0.0
        return 0.0


@dataclass
class FederatedUpdateMessage:
    """Message for federated parameter updates between peers.

    Attributes:
        sender_id: ID of the sending peer
        model_version: Target model version
        leaf_parameters: List of updated leaf parameters
        timestamp: When this message was created
        total_keys_trained: Total keys used across all updated leaves
    """
    sender_id: str
    model_version: int
    leaf_parameters: List[LeafModelParameters]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    total_keys_trained: int = 0

    def get_byte_size(self) -> int:
        """Calculate message size for network overhead analysis.

        Returns:
            Approximate size in bytes
        """
        # Header overhead + sum of parameter sizes
        header_size = 64  # sender_id, version, timestamp, etc.
        params_size = sum(len(p.to_bytes()) for p in self.leaf_parameters)
        return header_size + params_size


class FederatedRecursiveModel:
    """Federated Recursive Model for distributed DLHT training.

    This class implements the FRM algorithm from LEAD paper Section III-E3,
    enabling decentralized model updates where peers collaboratively refine
    leaf models without a permanent central coordinator.

    The algorithm works as follows:
    1. Each peer monitors proportion of new keys observed locally
    2. When new keys exceed threshold (40%), peer flags readiness
    3. When majority of neighbors (90%) are ready, peer becomes transient coordinator
    4. Transient coordinator aggregates leaf parameters from ready neighbors
    5. New model version is broadcast for network-wide adoption

    Attributes:
        num_leaf_models: Number of leaf models in the RMI
        model_type: Type of leaf model ('linear' or 'cubic')
        update_threshold: Proportion of new keys to trigger update (default 0.4)
        neighbor_ready_threshold: Proportion of neighbors ready (default 0.9)
        version: Current model version
        ready_for_update: Whether this peer is ready for federated update
    """

    def __init__(
        self,
        num_leaf_models: int,
        model_type: str = 'linear',
        update_threshold: float = 0.4,
        neighbor_ready_threshold: float = 0.9,
        stage1_coefficients: Optional[NDArray[np.float64]] = None
    ):
        """Initialize Federated Recursive Model.

        Args:
            num_leaf_models: Number of leaf models in RMI
            model_type: 'linear' or 'cubic' for leaf models
            update_threshold: Proportion of new keys to trigger update (0.0-1.0)
            neighbor_ready_threshold: Proportion of neighbors that must be ready (0.0-1.0)
            stage1_coefficients: Optional pre-trained stage 1 model coefficients

        Raises:
            ValueError: If parameters are invalid
        """
        if num_leaf_models < 1:
            raise ValueError(f"num_leaf_models must be >= 1, got {num_leaf_models}")
        if not 0 < update_threshold <= 1.0:
            raise ValueError(f"update_threshold must be in (0, 1], got {update_threshold}")
        if not 0 < neighbor_ready_threshold <= 1.0:
            raise ValueError(f"neighbor_ready_threshold must be in (0, 1], got {neighbor_ready_threshold}")
        if model_type not in ('linear', 'cubic'):
            raise ValueError(f"model_type must be 'linear' or 'cubic', got {model_type}")

        self.num_leaf_models = num_leaf_models
        self.model_type = ModelType(model_type)
        self.update_threshold = update_threshold
        self.neighbor_ready_threshold = neighbor_ready_threshold

        # Model state
        self._version = 0
        self._ready_for_update = False
        self._lock = threading.Lock()

        # Stage 1 model (L0 layer) - remains stable
        self._stage1_coefficients = (
            stage1_coefficients if stage1_coefficients is not None
            else np.array([0.0, 1.0])  # Default: linear identity
        )

        # Leaf models (L1 layer) - updated federally
        self._leaf_models: Dict[int, LeafModelParameters] = {}
        self._initialize_leaf_models()

        # Local training state
        self._keys_trained_since_update: Dict[int, int] = defaultdict(int)
        self._total_keys_trained = 0
        self._keys_at_last_update = 0

        # Neighbor coordination state
        self._neighbor_ready_status: Dict[str, bool] = {}
        self._neighbor_versions: Dict[str, int] = {}

        # Aggregated parameters (stored after aggregation)
        self._aggregated_parameters: Dict[int, LeafModelParameters] = {}

        # Performance metrics
        self.metrics = FRMMetrics()

        logger.info(
            f"Initialized FRM with {num_leaf_models} {model_type} leaf models, "
            f"update_threshold={update_threshold:.1%}, "
            f"neighbor_threshold={neighbor_ready_threshold:.1%}"
        )

    @property
    def model_version(self) -> int:
        """Get current model version."""
        return self._version

    @property
    def local_ready(self) -> bool:
        """Get whether this peer is ready for federated update."""
        return self._ready_for_update

    @local_ready.setter
    def local_ready(self, value: bool):
        """Set whether this peer is ready for federated update."""
        self._ready_for_update = value

    def _initialize_leaf_models(self) -> None:
        """Initialize leaf models with default parameters."""
        for i in range(self.num_leaf_models):
            # Default to identity mapping: predict position = index
            if self.model_type == ModelType.LINEAR:
                coefficients = np.array([1.0, float(i)])  # slope=1, intercept=i
            else:  # CUBIC
                coefficients = np.array([0.0, 0.0, 1.0, float(i)])  # cubic + linear

            self._leaf_models[i] = LeafModelParameters(
                leaf_index=i,
                coefficients=coefficients,
                min_key=float(i),
                max_key=float(i + 1),
                num_samples=0,
                version=0
            )

    @property
    def version(self) -> int:
        """Current model version."""
        return self._version

    @property
    def ready_for_update(self) -> bool:
        """Whether this peer is ready for federated update."""
        return self._ready_for_update

    def update_leaf_model(
        self,
        leaf_index: int,
        keys: NDArray[np.float64],
        positions: NDArray[np.int64]
    ) -> None:
        """Update a specific leaf model with new training data.

        This method is called when new keys are added to the system. It refines
        the leaf model's coefficients using gradient descent on the new data.

        Args:
            leaf_index: Index of leaf model to update (0 to num_leaf_models-1)
            keys: Array of keys in the new data
            positions: Array of relative positions for each key

        Raises:
            ValueError: If leaf_index is invalid or arrays have mismatched lengths
            ModelException: If model training fails
        """
        if not 0 <= leaf_index < self.num_leaf_models:
            raise ValueError(
                f"Invalid leaf_index {leaf_index}, "
                f"must be in [0, {self.num_leaf_models})"
            )
        if len(keys) != len(positions):
            raise ValueError(
                f"keys and positions must have same length, "
                f"got {len(keys)} vs {len(positions)}"
            )
        if len(keys) == 0:
            return  # Nothing to train

        with self._lock:
            leaf = self._leaf_models[leaf_index]

            try:
                # Simple gradient descent update (can be replaced with more sophisticated methods)
                learning_rate = 0.01
                for key, pos in zip(keys, positions):
                    # Compute prediction
                    if self.model_type == ModelType.LINEAR:
                        pred = leaf.coefficients[0] * key + leaf.coefficients[1]
                    else:  # CUBIC
                        pred = (leaf.coefficients[0] * key**3 +
                                leaf.coefficients[1] * key**2 +
                                leaf.coefficients[2] * key +
                                leaf.coefficients[3])

                    # Compute error and gradient
                    error = pred - pos
                    if self.model_type == ModelType.LINEAR:
                        grad = np.array([error * key, error])
                    else:  # CUBIC
                        grad = np.array([
                            error * key**3,
                            error * key**2,
                            error * key,
                            error
                        ])

                    # Update coefficients
                    leaf.coefficients -= learning_rate * grad

                # Update metadata
                leaf.num_samples += len(keys)
                leaf.min_key = min(leaf.min_key, float(np.min(keys)))
                leaf.max_key = max(leaf.max_key, float(np.max(keys)))
                leaf.last_updated = datetime.now(timezone.utc)

                # Track new keys for update threshold
                self._keys_trained_since_update[leaf_index] += len(keys)
                self._total_keys_trained += len(keys)

                # Record metrics
                self.metrics.record_leaf_update()

                # Check if we should flag ready for update
                self._check_update_readiness()

            except Exception as e:
                raise ModelException(
                    f"Failed to update leaf model {leaf_index}: {e}"
                ) from e

    def _check_update_readiness(self) -> None:
        """Check if proportion of new keys exceeds threshold."""
        if self._total_keys_trained == 0:
            return

        new_key_proportion = (
            (self._total_keys_trained - self._keys_at_last_update) /
            self._total_keys_trained
        )

        if new_key_proportion >= self.update_threshold:
            self._ready_for_update = True
            logger.info(
                f"Ready for federated update: {new_key_proportion:.1%} new keys "
                f"(threshold: {self.update_threshold:.1%})"
            )

    def should_trigger_federated_update(
        self,
        neighbor_ready_status: Optional[Dict[str, bool]] = None
    ) -> bool:
        """Check if this peer should become transient coordinator.

        A peer becomes transient coordinator when:
        1. It is ready for update (>40% new keys)
        2. Majority of neighbors (>90%) are also ready

        Args:
            neighbor_ready_status: Dict mapping neighbor_id -> ready status

        Returns:
            True if this peer should trigger federated update
        """
        if not self._ready_for_update:
            return False

        if neighbor_ready_status:
            self._neighbor_ready_status = neighbor_ready_status

        if not self._neighbor_ready_status:
            # No neighbors known yet, wait
            return False

        ready_count = sum(1 for ready in self._neighbor_ready_status.values() if ready)
        total_neighbors = len(self._neighbor_ready_status)

        if total_neighbors == 0:
            return False

        ready_proportion = ready_count / total_neighbors

        if ready_proportion >= self.neighbor_ready_threshold:
            logger.info(
                f"Triggering federated update: {ready_count}/{total_neighbors} "
                f"neighbors ready ({ready_proportion:.1%} >= "
                f"{self.neighbor_ready_threshold:.1%})"
            )
            return True

        return False

    def get_leaf_parameters(
        self,
        only_updated: bool = True
    ) -> List[LeafModelParameters]:
        """Get leaf model parameters for transfer to coordinator.

        Args:
            only_updated: If True, only return leaves updated since last federated update

        Returns:
            List of LeafModelParameters
        """
        with self._lock:
            if only_updated:
                # Only return leaves with new training data
                updated_indices = [
                    idx for idx, count in self._keys_trained_since_update.items()
                    if count > 0
                ]
                params = [self._leaf_models[idx] for idx in updated_indices]
            else:
                params = list(self._leaf_models.values())

            logger.debug(
                f"Exporting {len(params)} leaf parameters "
                f"(only_updated={only_updated})"
            )
            return params

    def aggregate_leaf_parameters(
        self,
        peer_parameters: List[Dict[str, Any]]
    ) -> Dict[int, LeafModelParameters]:
        """Aggregate leaf parameters from multiple peers via averaging.

        This implements the FedAvg algorithm: simple averaging of model parameters
        across peers. Each leaf model is averaged independently.

        Args:
            peer_parameters: List of dicts with keys:
                - sender_id: ID of the peer
                - model_version: Model version from peer
                - leaf_parameters: List of LeafModelParameters dicts
                - total_keys_trained: Total number of keys trained

        Returns:
            Dict mapping leaf_index -> aggregated LeafModelParameters
        """
        if not peer_parameters:
            return {}

        # Group parameters by leaf index
        leaf_groups: Dict[int, List[LeafModelParameters]] = defaultdict(list)
        for peer_data in peer_parameters:
            sender_id = peer_data['sender_id']
            leaf_params_dicts = peer_data['leaf_parameters']

            # Convert dicts to LeafModelParameters if needed
            for param_dict in leaf_params_dicts:
                if isinstance(param_dict, dict):
                    param = LeafModelParameters.from_dict(param_dict)
                else:
                    param = param_dict
                leaf_groups[param.leaf_index].append(param)

        # Average each leaf model
        aggregated = {}
        for leaf_index, params in leaf_groups.items():
            if not params:
                continue

            # Average coefficients
            all_coeffs = np.array([p.coefficients for p in params])
            avg_coefficients = np.mean(all_coeffs, axis=0)

            # Use min/max bounds across all peers
            min_key = min(p.min_key for p in params)
            max_key = max(p.max_key for p in params)
            total_samples = sum(p.num_samples for p in params)
            max_version = max(p.version for p in params)

            aggregated[leaf_index] = LeafModelParameters(
                leaf_index=leaf_index,
                coefficients=avg_coefficients,
                min_key=min_key,
                max_key=max_key,
                num_samples=total_samples,
                version=max_version + 1  # Increment version
            )

        # Store aggregated parameters
        with self._lock:
            self._aggregated_parameters = aggregated

        logger.info(
            f"Aggregated {len(aggregated)} leaf models from "
            f"{len(peer_parameters)} peers"
        )
        return aggregated

    def get_aggregated_parameters(self) -> List[LeafModelParameters]:
        """Get the most recently aggregated parameters.

        Returns:
            List of aggregated leaf model parameters
        """
        with self._lock:
            return list(self._aggregated_parameters.values())

    def apply_aggregated_parameters(
        self,
        aggregated_params: Dict[int, LeafModelParameters]
    ) -> None:
        """Apply aggregated parameters and increment model version.

        Args:
            aggregated_params: Dict of aggregated leaf parameters
        """
        with self._lock:
            for leaf_index, params in aggregated_params.items():
                if leaf_index in self._leaf_models:
                    self._leaf_models[leaf_index] = params

            # Increment global version
            self._version += 1

            # Reset training counters
            self._keys_at_last_update = self._total_keys_trained
            self._keys_trained_since_update.clear()
            self._ready_for_update = False

            logger.info(
                f"Applied {len(aggregated_params)} aggregated leaf models, "
                f"version now {self._version}"
            )

    def update_neighbor_status(
        self,
        neighbor_id: str,
        ready: bool,
        version: Optional[int] = None
    ) -> None:
        """Update neighbor's readiness status from heartbeat.

        Args:
            neighbor_id: ID of the neighbor peer
            ready: Whether neighbor is ready for update
            version: Neighbor's current model version
        """
        self._neighbor_ready_status[neighbor_id] = ready
        if version is not None:
            self._neighbor_versions[neighbor_id] = version

    def get_update_status(self) -> Dict[str, Any]:
        """Get current FRM status for monitoring.

        Returns:
            Dict with status information
        """
        with self._lock:
            new_key_proportion = 0.0
            if self._total_keys_trained > 0:
                new_key_proportion = (
                    (self._total_keys_trained - self._keys_at_last_update) /
                    self._total_keys_trained
                )

            ready_neighbors = sum(
                1 for ready in self._neighbor_ready_status.values() if ready
            )
            total_neighbors = len(self._neighbor_ready_status)

            return {
                'version': self._version,
                'ready_for_update': self._ready_for_update,
                'new_key_proportion': new_key_proportion,
                'update_threshold': self.update_threshold,
                'total_keys_trained': self._total_keys_trained,
                'updated_leaf_count': len([
                    idx for idx, count in self._keys_trained_since_update.items()
                    if count > 0
                ]),
                'neighbors_ready': ready_neighbors,
                'neighbors_total': total_neighbors,
                'neighbor_ready_proportion': (
                    ready_neighbors / total_neighbors if total_neighbors > 0 else 0.0
                )
            }

    def reset_ready_status(self) -> None:
        """Reset ready-for-update flag after update completion."""
        with self._lock:
            self._ready_for_update = False
            self._neighbor_ready_status.clear()

    def __repr__(self) -> str:
        return (
            f"FederatedRecursiveModel(leaves={self.num_leaf_models}, "
            f"type={self.model_type.value}, version={self.version}, "
            f"ready={self.ready_for_update})"
        )


def federated_average_models(
    models: List[FederatedRecursiveModel],
    target_version: Optional[int] = None
) -> FederatedRecursiveModel:
    """Average multiple FRM instances (for testing/simulation).

    This is a convenience function for testing federated averaging in
    simulation environments.

    Args:
        models: List of FRM instances to average
        target_version: Optional version number for result

    Returns:
        New FRM with averaged parameters

    Raises:
        ValueError: If models list is empty or models are incompatible
    """
    if not models:
        raise ValueError("Cannot average empty list of models")

    # Verify all models are compatible
    first = models[0]
    for model in models[1:]:
        if model.num_leaf_models != first.num_leaf_models:
            raise ValueError(
                f"Incompatible models: {model.num_leaf_models} vs "
                f"{first.num_leaf_models} leaf models"
            )
        if model.model_type != first.model_type:
            raise ValueError(
                f"Incompatible model types: {model.model_type} vs "
                f"{first.model_type}"
            )

    # Create new FRM
    result = FederatedRecursiveModel(
        num_leaf_models=first.num_leaf_models,
        model_type=first.model_type.value,
        update_threshold=first.update_threshold,
        neighbor_ready_threshold=first.neighbor_ready_threshold
    )

    # Collect all leaf parameters
    peer_params = [
        (f"peer-{i}", model.get_leaf_parameters(only_updated=False))
        for i, model in enumerate(models)
    ]

    # Aggregate and apply
    aggregated = result.aggregate_leaf_parameters(peer_params)
    result.apply_aggregated_parameters(aggregated)

    if target_version is not None:
        result._version = target_version

    return result
