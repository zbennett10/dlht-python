"""
Learned models for LEAD DHT (Recursive Model Index implementation)
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class LinearModel:
    """Simple linear regression model for RMI leaf nodes"""

    def __init__(self):
        self.slope = 0.0
        self.intercept = 0.0
        self.offset = 0.0  # Anchor offset for dynamic adjustment
        self.scale = 1.0   # Anchor scale for dynamic adjustment
        self.min_key = 0.0
        self.max_key = 1.0
        self.max_error = 0.0  # Maximum prediction error (error bound)
        
    def train(self, keys: np.ndarray, positions: np.ndarray):
        """Train linear model on keys and their positions"""
        if len(keys) == 0:
            return

        # Store normalization parameters
        self.min_key = np.min(keys)
        self.max_key = np.max(keys)

        if self.max_key == self.min_key:
            self.slope = 0.0
            self.intercept = 0.5
            self.max_error = 0.0
            return

        normalized_keys = (keys - self.min_key) / (self.max_key - self.min_key)

        # Linear regression
        n = len(keys)
        sum_x = np.sum(normalized_keys)
        sum_y = np.sum(positions)
        sum_xy = np.sum(normalized_keys * positions)
        sum_x2 = np.sum(normalized_keys ** 2)

        denominator = n * sum_x2 - sum_x ** 2
        if abs(denominator) < 1e-10:
            self.slope = 0.0
            self.intercept = 0.5
        else:
            self.slope = (n * sum_xy - sum_x * sum_y) / denominator
            self.intercept = (sum_y - self.slope * sum_x) / n

        # Calculate max prediction error (error bound)
        predictions = self.slope * normalized_keys + self.intercept
        errors = np.abs(predictions - positions)
        self.max_error = np.max(errors) if len(errors) > 0 else 0.0
        
    def predict(self, key: float) -> float:
        """Predict position for a key"""
        # Normalize key
        if self.max_key == self.min_key:
            normalized = 0.5
        else:
            normalized = (key - self.min_key) / (self.max_key - self.min_key)
            normalized = np.clip(normalized, 0, 1)
        
        prediction = self.slope * normalized + self.intercept
        # Apply anchor adjustments
        return self.offset + self.scale * prediction
        
    def update_anchor(self, keys: np.ndarray):
        """Update anchor parameters based on key distribution"""
        if len(keys) == 0:
            return
            
        predictions = np.array([self.predict(k) for k in keys])
        
        # Adjust offset to center median
        median_pred = np.median(predictions)
        self.offset -= (median_pred - 0.5)
        
        # Adjust scale so 95th percentile is near 1.0
        p95 = np.percentile(predictions, 95)
        if p95 > 0.1:  # Avoid division by very small numbers
            self.scale *= 0.95 / p95
            
    def to_dict(self) -> dict:
        """Serialize model parameters"""
        return {
            'type': 'linear',
            'slope': float(self.slope),
            'intercept': float(self.intercept),
            'offset': float(self.offset),
            'scale': float(self.scale),
            'min_key': float(self.min_key),
            'max_key': float(self.max_key),
            'max_error': float(self.max_error)
        }
        
    @classmethod
    def from_dict(cls, params: dict) -> 'LinearModel':
        """Deserialize model parameters"""
        model = cls()
        model.slope = params['slope']
        model.intercept = params['intercept']
        model.offset = params['offset']
        model.scale = params['scale']
        model.min_key = params['min_key']
        model.max_key = params['max_key']
        model.max_error = params.get('max_error', 0.0)  # Backwards compatibility
        return model


class CubicModel:
    """Cubic polynomial model for RMI leaf nodes"""

    def __init__(self):
        self.coeffs = np.zeros(4)  # a*x^3 + b*x^2 + c*x + d
        self.offset = 0.0
        self.scale = 1.0
        self.min_key = 0.0
        self.max_key = 1.0
        self.max_error = 0.0  # Maximum prediction error (error bound)
        
    def train(self, keys: np.ndarray, positions: np.ndarray):
        """Train cubic model using polynomial fitting"""
        if len(keys) < 4:  # Need at least 4 points for cubic
            self.coeffs = np.array([0, 0, 0, 0.5])
            self.max_error = 0.0
            return

        self.min_key = np.min(keys)
        self.max_key = np.max(keys)

        if self.max_key == self.min_key:
            self.coeffs = np.array([0, 0, 0, 0.5])
            self.max_error = 0.0
            return

        normalized_keys = (keys - self.min_key) / (self.max_key - self.min_key)
        self.coeffs = np.polyfit(normalized_keys, positions, 3)

        # Calculate max prediction error (error bound)
        predictions = np.polyval(self.coeffs, normalized_keys)
        errors = np.abs(predictions - positions)
        self.max_error = np.max(errors) if len(errors) > 0 else 0.0
        
    def predict(self, key: float) -> float:
        """Predict position for a key"""
        if self.max_key == self.min_key:
            normalized = 0.5
        else:
            normalized = (key - self.min_key) / (self.max_key - self.min_key)
            normalized = np.clip(normalized, 0, 1)
        
        prediction = np.polyval(self.coeffs, normalized)
        return self.offset + self.scale * prediction
        
    def update_anchor(self, keys: np.ndarray):
        """Update anchor parameters"""
        if len(keys) == 0:
            return
            
        predictions = np.array([self.predict(k) for k in keys])
        median_pred = np.median(predictions)
        self.offset -= (median_pred - 0.5)
        
        p95 = np.percentile(predictions, 95)
        if p95 > 0.1:
            self.scale *= 0.95 / p95
            
    def to_dict(self) -> dict:
        """Serialize model parameters"""
        return {
            'type': 'cubic',
            'coeffs': self.coeffs.tolist(),
            'offset': float(self.offset),
            'scale': float(self.scale),
            'min_key': float(self.min_key),
            'max_key': float(self.max_key),
            'max_error': float(self.max_error)
        }

    @classmethod
    def from_dict(cls, params: dict) -> 'CubicModel':
        """Deserialize model parameters"""
        model = cls()
        model.coeffs = np.array(params['coeffs'])
        model.offset = params['offset']
        model.scale = params['scale']
        model.min_key = params['min_key']
        model.max_key = params['max_key']
        model.max_error = params.get('max_error', 0.0)  # Backwards compatibility
        return model


class RecursiveModelIndex:
    """Two-stage Recursive Model Index for learned hashing"""

    def __init__(self, branching_factor: int = 100, model_type: str = 'linear',
                 hash_space_size: int = 2**160, stage1_model_type: str = 'linear'):
        self.branching_factor = branching_factor
        self.model_type = model_type  # Type for stage 2 (leaf) models
        self.stage1_model_type = stage1_model_type  # Type for stage 1 (root) model
        self.hash_space_size = hash_space_size
        self.version = 0

        # Store training data statistics for normalization
        self.training_data_size = 0
        self.min_training_key = 0.0
        self.max_training_key = float(hash_space_size)

        # Initialize stage 1 model (can be linear or cubic)
        if stage1_model_type == 'cubic':
            self.stage1_model = CubicModel()
        else:
            self.stage1_model = LinearModel()

        # Initialize stage 2 (leaf) models
        self.stage2_models = []
        for _ in range(branching_factor):
            if model_type == 'cubic':
                self.stage2_models.append(CubicModel())
            else:
                self.stage2_models.append(LinearModel())
                
    def train(self, keys: np.ndarray):
        """Train the RMI on a dataset of keys"""
        if len(keys) == 0:
            logger.warning("Training RMI with empty dataset")
            return

        # Sort keys
        sorted_keys = np.sort(keys)
        n = len(sorted_keys)

        # Store training statistics
        self.training_data_size = n
        self.min_training_key = float(np.min(sorted_keys))
        self.max_training_key = float(np.max(sorted_keys))

        if self.max_training_key == self.min_training_key:
            logger.warning("All keys are identical, using default model")
            return

        # Normalize keys to [0, 1]
        normalized_keys = (sorted_keys - self.min_training_key) / (self.max_training_key - self.min_training_key)
        positions = np.arange(n) / n

        # Train stage 1 model to predict bucket
        bucket_ids = np.floor(positions * self.branching_factor).astype(int)
        bucket_ids = np.clip(bucket_ids, 0, self.branching_factor - 1)
        self.stage1_model.train(normalized_keys, bucket_ids)

        # Train stage 2 models
        for bucket_id in range(self.branching_factor):
            mask = bucket_ids == bucket_id
            if np.sum(mask) > 0:
                bucket_keys = normalized_keys[mask]
                bucket_positions = (positions[mask] - bucket_id / self.branching_factor) * self.branching_factor
                self.stage2_models[bucket_id].train(bucket_keys, bucket_positions)

        self.version += 1
        logger.info(f"RMI trained on {n} keys, version {self.version}")
        
    def predict(self, key: float) -> int:
        """Predict hash value for a key"""
        # Normalize key using same range as training data
        if self.max_training_key == self.min_training_key:
            # No training or all keys identical - map to middle of hash space
            normalized_key = 0.5
        else:
            # Clip to training range and normalize
            clipped_key = np.clip(key, self.min_training_key, self.max_training_key)
            normalized_key = (clipped_key - self.min_training_key) / (self.max_training_key - self.min_training_key)

        # Stage 1: predict bucket
        bucket_prediction = self.stage1_model.predict(normalized_key)
        bucket_id = int(np.clip(bucket_prediction, 0, self.branching_factor - 1))

        # Stage 2: predict position within bucket
        position_in_bucket = self.stage2_models[bucket_id].predict(normalized_key)

        # Convert to hash value
        overall_position = (bucket_id + np.clip(position_in_bucket, 0, 1)) / self.branching_factor
        hash_value = int(overall_position * self.hash_space_size)

        return max(0, min(hash_value, self.hash_space_size - 1))

    def predict_with_error(self, key: float) -> tuple[int, float]:
        """Predict hash value for a key with error bound"""
        # Normalize key using same range as training data
        if self.max_training_key == self.min_training_key:
            normalized_key = 0.5
        else:
            clipped_key = np.clip(key, self.min_training_key, self.max_training_key)
            normalized_key = (clipped_key - self.min_training_key) / (self.max_training_key - self.min_training_key)

        # Stage 1: predict bucket
        bucket_prediction = self.stage1_model.predict(normalized_key)
        bucket_id = int(np.clip(bucket_prediction, 0, self.branching_factor - 1))

        # Stage 2: predict position within bucket
        position_in_bucket = self.stage2_models[bucket_id].predict(normalized_key)

        # Get error bounds
        stage1_error = self.stage1_model.max_error
        stage2_error = self.stage2_models[bucket_id].max_error

        # Combined error (conservative estimate)
        combined_error = stage1_error + stage2_error

        # Convert to hash value
        overall_position = (bucket_id + np.clip(position_in_bucket, 0, 1)) / self.branching_factor
        hash_value = int(overall_position * self.hash_space_size)

        # Convert error to hash space
        error_in_hash_space = combined_error * self.hash_space_size

        return max(0, min(hash_value, self.hash_space_size - 1)), error_in_hash_space

    def predict_range(self, key_min: float, key_max: float) -> tuple[int, int]:
        """
        Predict hash range for keys in [key_min, key_max].

        Returns conservative bounds [hash_min, hash_max] that cover all possible
        hash values for keys in the input range, accounting for model error.

        This enables range queries by identifying which hash space regions
        might contain relevant keys.

        Args:
            key_min: Minimum key value in range
            key_max: Maximum key value in range

        Returns:
            (hash_min, hash_max) tuple covering the key range
        """
        if key_min > key_max:
            key_min, key_max = key_max, key_min

        # Predict with error bounds for both endpoints
        hash_min_pred, error_min = self.predict_with_error(key_min)
        hash_max_pred, error_max = self.predict_with_error(key_max)

        # Expand range by error bounds (conservative)
        hash_min = int(max(0, hash_min_pred - error_min))
        hash_max = int(min(self.hash_space_size - 1, hash_max_pred + error_max))

        return (hash_min, hash_max)

    def predict_range_detailed(self, key_min: float, key_max: float,
                               num_samples: int = 10) -> tuple[int, int, list]:
        """
        Predict hash range with detailed sampling across the key range.

        More accurate than predict_range() for large ranges by sampling
        multiple points and checking for non-monotonic behavior in the
        learned hash function.

        Args:
            key_min: Minimum key value
            key_max: Maximum key value
            num_samples: Number of sample points to check

        Returns:
            (hash_min, hash_max, sample_hashes) tuple
        """
        if key_min > key_max:
            key_min, key_max = key_max, key_min

        # Sample keys uniformly across the range
        if num_samples < 2:
            num_samples = 2

        sample_keys = np.linspace(key_min, key_max, num_samples)
        sample_hashes = []

        for key in sample_keys:
            hash_val, error = self.predict_with_error(key)
            # Include error bounds for each sample
            sample_hashes.append(int(max(0, hash_val - error)))
            sample_hashes.append(int(min(self.hash_space_size - 1, hash_val + error)))

        hash_min = min(sample_hashes)
        hash_max = max(sample_hashes)

        return (hash_min, hash_max, sample_hashes)

    def update_leaf(self, bucket_id: int, keys: np.ndarray, positions: np.ndarray):
        """Update a specific leaf model (for federated updates)"""
        if 0 <= bucket_id < self.branching_factor and len(keys) > 0:
            self.stage2_models[bucket_id].train(keys, positions)
            
    def get_leaf_params(self, bucket_id: int) -> dict:
        """Get parameters of a leaf model for federated updates"""
        if 0 <= bucket_id < self.branching_factor:
            return self.stage2_models[bucket_id].to_dict()
        return {}
        
    def set_leaf_params(self, bucket_id: int, params: dict):
        """Set parameters of a leaf model"""
        if 0 <= bucket_id < self.branching_factor:
            if params['type'] == 'linear':
                self.stage2_models[bucket_id] = LinearModel.from_dict(params)
            elif params['type'] == 'cubic':
                self.stage2_models[bucket_id] = CubicModel.from_dict(params)
                
    def to_dict(self) -> dict:
        """Serialize entire model"""
        return {
            'version': self.version,
            'branching_factor': self.branching_factor,
            'model_type': self.model_type,
            'stage1_model_type': self.stage1_model_type,
            'hash_space_size': self.hash_space_size,
            'training_data_size': self.training_data_size,
            'min_training_key': self.min_training_key,
            'max_training_key': self.max_training_key,
            'stage1_model': self.stage1_model.to_dict(),
            'stage2_models': [m.to_dict() for m in self.stage2_models]
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'RecursiveModelIndex':
        """Deserialize model"""
        stage1_model_type = data.get('stage1_model_type', 'linear')  # Backwards compatibility
        rmi = cls(
            branching_factor=data['branching_factor'],
            model_type=data['model_type'],
            hash_space_size=data['hash_space_size'],
            stage1_model_type=stage1_model_type
        )
        rmi.version = data['version']

        # Restore training statistics (with backwards compatibility)
        rmi.training_data_size = data.get('training_data_size', 0)
        rmi.min_training_key = data.get('min_training_key', 0.0)
        rmi.max_training_key = data.get('max_training_key', float(data['hash_space_size']))

        # Deserialize stage 1 model
        stage1_params = data['stage1_model']
        if stage1_params['type'] == 'cubic':
            rmi.stage1_model = CubicModel.from_dict(stage1_params)
        else:
            rmi.stage1_model = LinearModel.from_dict(stage1_params)

        # Deserialize stage 2 models
        rmi.stage2_models = []
        for params in data['stage2_models']:
            if params['type'] == 'linear':
                rmi.stage2_models.append(LinearModel.from_dict(params))
            elif params['type'] == 'cubic':
                rmi.stage2_models.append(CubicModel.from_dict(params))

        return rmi