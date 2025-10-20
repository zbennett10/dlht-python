"""
Adaptive training strategy for DLHT Recursive Model Index.

Implements modern ML practices for distributed learned hash tables:
- Trigger-based retraining (data drift, error thresholds, volume)
- Incremental/online learning for continuous improvement
- Federated model averaging across registries
- Performance monitoring and metrics
"""

import numpy as np
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Metrics tracking for RMI model performance"""

    # Training metadata
    last_training_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    training_count: int = 0
    training_data_size: int = 0
    model_version: int = 0

    # Prediction accuracy metrics
    total_predictions: int = 0
    total_absolute_error: float = 0.0
    mean_absolute_error: float = 0.0
    max_error_observed: float = 0.0

    # Data drift metrics
    key_min_observed: float = 0.0
    key_max_observed: float = 0.0
    key_mean: float = 0.0
    key_std: float = 0.0

    # Recent error samples for drift detection
    recent_errors: deque = field(default_factory=lambda: deque(maxlen=1000))

    # Model staleness
    keys_added_since_training: int = 0
    time_since_training_seconds: float = 0.0

    def update_prediction_error(self, predicted: int, actual: int):
        """Record a prediction error"""
        error = abs(predicted - actual)
        self.total_predictions += 1
        self.total_absolute_error += error
        self.mean_absolute_error = self.total_absolute_error / self.total_predictions
        self.max_error_observed = max(self.max_error_observed, error)
        self.recent_errors.append(error)

    def update_staleness(self):
        """Update model staleness metrics"""
        now = datetime.now(timezone.utc)
        self.time_since_training_seconds = (now - self.last_training_time).total_seconds()

    def reset_after_training(self, data_size: int, version: int):
        """Reset metrics after successful retraining"""
        self.last_training_time = datetime.now(timezone.utc)
        self.training_count += 1
        self.training_data_size = data_size
        self.model_version = version
        self.keys_added_since_training = 0
        self.total_predictions = 0
        self.total_absolute_error = 0.0
        self.mean_absolute_error = 0.0
        self.recent_errors.clear()

    def get_recent_error_trend(self) -> float:
        """
        Calculate error trend: positive = increasing errors (model degrading)

        Compares first half vs second half of recent errors
        """
        if len(self.recent_errors) < 100:
            return 0.0

        errors_list = list(self.recent_errors)
        mid = len(errors_list) // 2
        first_half_mean = np.mean(errors_list[:mid])
        second_half_mean = np.mean(errors_list[mid:])

        # Return relative change
        if first_half_mean > 0:
            return (second_half_mean - first_half_mean) / first_half_mean
        return 0.0

    def to_dict(self) -> dict:
        """Serialize metrics for monitoring/logging"""
        return {
            'last_training_time': self.last_training_time.isoformat(),
            'training_count': self.training_count,
            'training_data_size': self.training_data_size,
            'model_version': self.model_version,
            'total_predictions': self.total_predictions,
            'mean_absolute_error': self.mean_absolute_error,
            'max_error_observed': self.max_error_observed,
            'keys_added_since_training': self.keys_added_since_training,
            'time_since_training_hours': self.time_since_training_seconds / 3600,
            'error_trend': self.get_recent_error_trend()
        }


@dataclass
class TrainingConfig:
    """Configuration for adaptive training strategy"""

    # Error-based triggers
    enable_error_trigger: bool = True
    max_mean_error_threshold: float = 0.05  # Retrain if MAE > 5% of hash space
    error_trend_threshold: float = 0.2  # Retrain if error increasing by 20%

    # Volume-based triggers
    enable_volume_trigger: bool = True
    min_keys_for_training: int = 100  # Don't train with too few keys
    new_keys_trigger_count: int = 1000  # Retrain after N new keys
    new_keys_trigger_percent: float = 0.1  # Or after X% growth

    # Time-based triggers
    enable_staleness_trigger: bool = True
    max_staleness_hours: float = 24.0  # Retrain if model older than 24h

    # Incremental learning
    enable_incremental_learning: bool = True
    incremental_update_frequency: int = 100  # Update after N new keys

    # Federated learning
    enable_federated_averaging: bool = True
    federated_sync_interval_seconds: float = 3600.0  # Sync every hour

    # Reservoir sampling for efficient retraining
    reservoir_sample_size: int = 10000  # Keep up to 10k representative samples


class AdaptiveTrainingManager:
    """
    Manages adaptive training for DLHT RMI models.

    Responsibilities:
    - Monitor model performance and data drift
    - Trigger retraining based on multiple conditions
    - Support incremental learning for continuous improvement
    - Coordinate federated model averaging across registries
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.metrics = TrainingMetrics()

        # Reservoir sampling buffer for efficient retraining
        self.reservoir: List[float] = []
        self.reservoir_count = 0  # Total keys seen

        # Last federated sync time
        self.last_federated_sync = datetime.now(timezone.utc)

    def should_retrain(self, force: bool = False) -> Tuple[bool, str]:
        """
        Determine if model should be retrained.

        Args:
            force: Force retraining regardless of conditions

        Returns:
            (should_retrain, reason) tuple
        """
        if force:
            return (True, "forced retraining")

        self.metrics.update_staleness()

        # Error-based trigger
        if self.config.enable_error_trigger:
            if self.metrics.total_predictions > 100:  # Need enough samples
                if self.metrics.mean_absolute_error > self.config.max_mean_error_threshold:
                    return (True, f"high mean error: {self.metrics.mean_absolute_error:.4f}")

                error_trend = self.metrics.get_recent_error_trend()
                if error_trend > self.config.error_trend_threshold:
                    return (True, f"increasing error trend: +{error_trend*100:.1f}%")

        # Volume-based trigger
        if self.config.enable_volume_trigger:
            if self.metrics.keys_added_since_training >= self.config.new_keys_trigger_count:
                return (True, f"new keys threshold: {self.metrics.keys_added_since_training} keys")

            if self.metrics.training_data_size > 0:
                growth_percent = self.metrics.keys_added_since_training / self.metrics.training_data_size
                if growth_percent >= self.config.new_keys_trigger_percent:
                    return (True, f"growth threshold: +{growth_percent*100:.1f}%")

        # Time-based staleness trigger
        if self.config.enable_staleness_trigger:
            hours_since_training = self.metrics.time_since_training_seconds / 3600
            if hours_since_training >= self.config.max_staleness_hours:
                return (True, f"model staleness: {hours_since_training:.1f}h old")

        return (False, "no trigger conditions met")

    def on_key_inserted(self, key: float):
        """
        Called when a new key is inserted into the DHT.

        Updates reservoir sample and staleness metrics.
        """
        self.metrics.keys_added_since_training += 1

        # Reservoir sampling: maintain representative sample of all keys
        self.reservoir_count += 1
        if len(self.reservoir) < self.config.reservoir_sample_size:
            self.reservoir.append(key)
        else:
            # Randomly replace existing sample with probability k/n
            j = np.random.randint(0, self.reservoir_count)
            if j < self.config.reservoir_sample_size:
                self.reservoir[j] = key

    def on_prediction_made(self, key: float, predicted_hash: int, actual_hash: int):
        """
        Called when a hash prediction is made.

        Tracks prediction accuracy for drift detection.
        """
        self.metrics.update_prediction_error(predicted_hash, actual_hash)

    def get_training_data(self, all_keys: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get training data for retraining.

        Uses reservoir sample if available for efficiency, otherwise uses all keys.

        Args:
            all_keys: All keys currently in the DHT (fallback if reservoir empty)

        Returns:
            Sorted array of keys to train on
        """
        if len(self.reservoir) >= self.config.min_keys_for_training:
            logger.info(f"Using reservoir sample: {len(self.reservoir)} keys")
            return np.array(sorted(self.reservoir))
        elif all_keys is not None and len(all_keys) > 0:
            logger.info(f"Using all keys: {len(all_keys)} keys")
            # If reservoir is small, repopulate it from all_keys
            if len(all_keys) <= self.config.reservoir_sample_size:
                self.reservoir = list(all_keys)
            else:
                # Sample from all_keys
                indices = np.random.choice(len(all_keys), self.config.reservoir_sample_size, replace=False)
                self.reservoir = [all_keys[i] for i in indices]
            self.reservoir_count = len(all_keys)
            return np.array(sorted(all_keys))
        else:
            logger.warning("No training data available")
            return np.array([])

    def after_training(self, data_size: int, model_version: int):
        """
        Called after successful model retraining.

        Resets metrics and updates training history.
        """
        self.metrics.reset_after_training(data_size, model_version)
        logger.info(f"Training completed: version {model_version}, {data_size} keys")

    def should_sync_federated(self) -> bool:
        """Check if it's time to sync with other registries"""
        if not self.config.enable_federated_averaging:
            return False

        elapsed = (datetime.now(timezone.utc) - self.last_federated_sync).total_seconds()
        return elapsed >= self.config.federated_sync_interval_seconds

    def on_federated_sync(self):
        """Mark that federated sync occurred"""
        self.last_federated_sync = datetime.now(timezone.utc)

    def get_metrics(self) -> dict:
        """Get current training metrics for monitoring"""
        return self.metrics.to_dict()


def federated_average_models(models: List[Dict], weights: Optional[List[float]] = None) -> Dict:
    """
    Perform federated averaging of RMI models from multiple registries.

    This is a simplified version of FedAvg (McMahan et al., 2017) adapted
    for RMI linear/cubic models.

    Args:
        models: List of serialized RMI models (from to_dict())
        weights: Optional weights for each model (e.g., based on data size)
                Default: equal weighting

    Returns:
        Averaged model parameters (dict format)
    """
    if not models:
        raise ValueError("No models provided for averaging")

    if len(models) == 1:
        return models[0]

    # Equal weights if not provided
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    else:
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

    logger.info(f"Federated averaging {len(models)} models with weights {weights}")

    # Average stage 1 model parameters
    stage1_type = models[0]['stage1_model']['type']
    stage1_avg = {'type': stage1_type}

    if stage1_type == 'linear':
        stage1_avg['slope'] = sum(m['stage1_model']['slope'] * w for m, w in zip(models, weights))
        stage1_avg['intercept'] = sum(m['stage1_model']['intercept'] * w for m, w in zip(models, weights))
        stage1_avg['offset'] = sum(m['stage1_model'].get('offset', 0) * w for m, w in zip(models, weights))
        stage1_avg['scale'] = sum(m['stage1_model'].get('scale', 1) * w for m, w in zip(models, weights))
        stage1_avg['min_key'] = min(m['stage1_model']['min_key'] for m in models)
        stage1_avg['max_key'] = max(m['stage1_model']['max_key'] for m in models)
        stage1_avg['max_error'] = max(m['stage1_model'].get('max_error', 0) for m in models)
    elif stage1_type == 'cubic':
        # Average polynomial coefficients
        num_coeffs = len(models[0]['stage1_model']['coeffs'])
        avg_coeffs = []
        for i in range(num_coeffs):
            avg_coeffs.append(sum(m['stage1_model']['coeffs'][i] * w for m, w in zip(models, weights)))
        stage1_avg['coeffs'] = avg_coeffs
        stage1_avg['offset'] = sum(m['stage1_model'].get('offset', 0) * w for m, w in zip(models, weights))
        stage1_avg['scale'] = sum(m['stage1_model'].get('scale', 1) * w for m, w in zip(models, weights))
        stage1_avg['min_key'] = min(m['stage1_model']['min_key'] for m in models)
        stage1_avg['max_key'] = max(m['stage1_model']['max_key'] for m in models)
        stage1_avg['max_error'] = max(m['stage1_model'].get('max_error', 0) for m in models)

    # Average stage 2 models (leaf models)
    branching_factor = models[0]['branching_factor']
    stage2_avg = []

    for bucket_id in range(branching_factor):
        bucket_models = [m['stage2_models'][bucket_id] for m in models]
        bucket_type = bucket_models[0]['type']

        bucket_avg = {'type': bucket_type}

        if bucket_type == 'linear':
            bucket_avg['slope'] = sum(bm['slope'] * w for bm, w in zip(bucket_models, weights))
            bucket_avg['intercept'] = sum(bm['intercept'] * w for bm, w in zip(bucket_models, weights))
            bucket_avg['offset'] = sum(bm.get('offset', 0) * w for bm, w in zip(bucket_models, weights))
            bucket_avg['scale'] = sum(bm.get('scale', 1) * w for bm, w in zip(bucket_models, weights))
            bucket_avg['min_key'] = min(bm['min_key'] for bm in bucket_models)
            bucket_avg['max_key'] = max(bm['max_key'] for bm in bucket_models)
            bucket_avg['max_error'] = max(bm.get('max_error', 0) for bm in bucket_models)
        elif bucket_type == 'cubic':
            num_coeffs = len(bucket_models[0]['coeffs'])
            avg_coeffs = []
            for i in range(num_coeffs):
                avg_coeffs.append(sum(bm['coeffs'][i] * w for bm, w in zip(bucket_models, weights)))
            bucket_avg['coeffs'] = avg_coeffs
            bucket_avg['offset'] = sum(bm.get('offset', 0) * w for bm, w in zip(bucket_models, weights))
            bucket_avg['scale'] = sum(bm.get('scale', 1) * w for bm, w in zip(bucket_models, weights))
            bucket_avg['min_key'] = min(bm['min_key'] for bm in bucket_models)
            bucket_avg['max_key'] = max(bm['max_key'] for bm in bucket_models)
            bucket_avg['max_error'] = max(bm.get('max_error', 0) for bm in bucket_models)

        stage2_avg.append(bucket_avg)

    # Build averaged model
    averaged_model = {
        'version': max(m['version'] for m in models) + 1,  # Increment version
        'branching_factor': branching_factor,
        'model_type': models[0]['model_type'],
        'stage1_model_type': models[0].get('stage1_model_type', 'linear'),
        'hash_space_size': models[0]['hash_space_size'],
        'training_data_size': sum(m.get('training_data_size', 0) for m in models),
        'min_training_key': min(m.get('min_training_key', 0) for m in models),
        'max_training_key': max(m.get('max_training_key', 0) for m in models),
        'stage1_model': stage1_avg,
        'stage2_models': stage2_avg
    }

    logger.info(f"Federated averaging complete: version {averaged_model['version']}")

    return averaged_model
