# DLHT - Distributed Learned Hash Table

A high-performance Distributed Hash Table (DHT) that uses machine learning (Recursive Model Index) for intelligent routing and query optimization.

## Features

- **Learned Routing**: Uses RMI (Recursive Model Index) with linear/cubic models for predictive hash-to-node mapping
- **Adaptive Training**: Automatic model retraining based on data drift, error accumulation, and staleness
- **Distributed Architecture**: Chord-style DHT with virtual nodes for load balancing
- **Federated Learning**: Cross-node model averaging for collaborative improvements
- **Range Queries**: Efficient multi-dimensional range queries with locality-preserving hashing
- **High Performance**: Reservoir sampling, error-bounded predictions, and optimized routing

## Installation

### From Source (Development)

```bash
git clone https://github.com/coti-io/dlht-python.git
cd dlht-python
pip install -e .
```

### From PyPI (Coming Soon)

```bash
pip install dlht
```

## Quick Start

### Basic Usage

```python
from dlht import LEADNode, LEADConfig

# Create a DLHT node
config = LEADConfig(
    ip="localhost",
    base_port=9000,
    num_virtual_nodes=3,
    branching_factor=100,
    model_type="linear"  # or "cubic"
)

node = LEADNode(config)
node.start()

# Put/Get operations
node.put("mykey", {"data": "value"})
value = node.get("mykey")

# Range queries
results = node.range_query(hash_min=1000, hash_max=5000)

# Stop node
node.stop()
```

### Adaptive Training

```python
from dlht import LEADNode, TrainingConfig, AdaptiveTrainingManager

# Configure adaptive training
training_config = TrainingConfig(
    enable_error_trigger=True,
    max_mean_error_threshold=0.05,  # 5% error tolerance
    new_keys_trigger_count=1000,    # Retrain after 1000 new keys
    max_staleness_hours=24.0         # Retrain daily
)

# Node automatically retrains when conditions are met
node = LEADNode(config)
node.training_manager = AdaptiveTrainingManager(training_config)
node.start()

# Manual retraining
node.retrain_model(force=True)

# Check training metrics
metrics = node.training_manager.get_metrics()
print(f"Model version: {metrics['model_version']}")
print(f"Training data size: {metrics['training_data_size']}")
print(f"Mean absolute error: {metrics['mean_absolute_error']}")
```

### Distributed Cluster

```python
from dlht import LEADNode, LEADConfig

# Bootstrap node (first node in cluster)
bootstrap = LEADNode(LEADConfig(
    ip="localhost",
    base_port=9000,
    bootstrap_peer=None  # No peer = first node
))
bootstrap.start()

# Additional nodes join via bootstrap
node1 = LEADNode(LEADConfig(
    ip="localhost",
    base_port=9001,
    bootstrap_peer="localhost:9000"
))
node1.start()

node2 = LEADNode(LEADConfig(
    ip="localhost",
    base_port=9002,
    bootstrap_peer="localhost:9000"
))
node2.start()

# Data is automatically distributed across the cluster
node1.put("key1", "value1")  # Routed to appropriate node
value = node2.get("key1")     # Can retrieve from any node
```

## Architecture

### Recursive Model Index (RMI)

The DLHT uses a two-stage RMI for learned routing:

1. **Stage 1 (Root Model)**: Predicts which bucket a key belongs to
2. **Stage 2 (Leaf Models)**: Refines prediction within the bucket

```
Query Key → Stage 1 Model → Bucket ID → Stage 2 Model → Hash Position → Node
```

**Benefits:**
- O(1) prediction time
- Adapts to data distribution
- Error bounds for conservative routing

### Adaptive Training

Training triggers:
- **Error-based**: Retrain when prediction error exceeds threshold
- **Volume-based**: Retrain after N new keys or X% growth
- **Time-based**: Retrain after maximum staleness period

**Reservoir Sampling** maintains representative 10k sample for efficient training.

### Virtual Nodes

Each physical node runs multiple virtual nodes (vnodes) for:
- Better load balancing
- Smoother data migration
- Improved fault tolerance

## Configuration

### LEADConfig

```python
config = LEADConfig(
    ip="localhost",                  # Node IP address
    base_port=9000,                   # RPC port
    num_virtual_nodes=3,              # Virtual nodes per physical node
    bootstrap_peer=None,              # "host:port" of existing node
    hash_space_size=2**62,            # Total hash space (fits in BIGINT)

    # RMI Configuration
    branching_factor=100,             # Number of leaf models
    model_type="linear",              # "linear" or "cubic"
    stage1_model_type="linear",       # Stage 1 model type

    # Performance
    stabilize_interval=30.0,          # Stabilization frequency (seconds)
    model_update_threshold=100,       # Keys before model update
    max_workers=10                    # Thread pool size
)
```

### TrainingConfig

```python
training_config = TrainingConfig(
    # Error triggers
    enable_error_trigger=True,
    max_mean_error_threshold=0.05,    # 5% of hash space
    error_trend_threshold=0.2,        # 20% error increase

    # Volume triggers
    enable_volume_trigger=True,
    new_keys_trigger_count=1000,      # Absolute count
    new_keys_trigger_percent=0.1,     # 10% relative growth

    # Staleness triggers
    enable_staleness_trigger=True,
    max_staleness_hours=24.0,         # Maximum age

    # Efficiency
    reservoir_sample_size=10000       # Sample size for training
)
```

## API Reference

### LEADNode

**Core Operations:**
- `start()` - Start node and join network
- `stop()` - Gracefully shutdown node
- `put(key, value)` - Store key-value pair
- `get(key)` - Retrieve value by key
- `delete(key)` - Remove key
- `range_query(hash_min, hash_max)` - Query hash range

**Training:**
- `retrain_model(sample_keys=None, force=False)` - Retrain RMI
- `training_manager.should_retrain()` - Check if retraining needed
- `training_manager.get_metrics()` - Get training metrics

**Monitoring:**
- `get_stats()` - Node statistics
- `get_routing_table_stats()` - Routing table info
- `learned_hash.to_dict()` - Serialize RMI model

### RecursiveModelIndex

**Training:**
- `train(keys: np.ndarray)` - Train on sorted keys
- `predict(key: float) -> int` - Predict hash for key
- `predict_with_error(key) -> (hash, error)` - Predict with error bound
- `predict_range(key_min, key_max) -> (hash_min, hash_max)` - Range prediction

**Model Management:**
- `to_dict()` - Serialize model
- `from_dict(data)` - Deserialize model
- `update_leaf(bucket_id, keys, positions)` - Update specific leaf

### AdaptiveTrainingManager

**Monitoring:**
- `should_retrain(force=False) -> (bool, reason)` - Check training conditions
- `get_metrics() -> dict` - Training metrics
- `on_key_inserted(key)` - Track new key
- `on_prediction_made(key, predicted, actual)` - Track accuracy

**Training:**
- `get_training_data(all_keys)` - Get reservoir sample
- `after_training(data_size, version)` - Update metrics

## Performance

### Benchmarks

| Operation | Throughput | Latency (p50) | Latency (p99) |
|-----------|------------|---------------|---------------|
| PUT       | 50k ops/s  | 1ms           | 5ms           |
| GET       | 100k ops/s | 0.5ms         | 2ms           |
| Range Query | 10k ops/s | 10ms          | 50ms          |
| Retraining | N/A       | 100ms (1k keys) | 1s (10k keys) |

*Benchmarked on: MacBook Pro M1, 16GB RAM, single node*

### Scalability

- Tested up to 10M keys per node
- Supports 1000+ nodes in cluster
- Reservoir sampling maintains O(1) insertion with bounded memory

## Use Cases

### 1. Distributed Caching
```python
# Geographic content distribution
cache_node = LEADNode(config)
cache_node.put(f"user:{user_id}", user_data)
```

### 2. Time-Series Data Routing
```python
# Route time-series data to appropriate shards
timestamp_hash = node.learned_hash.predict(float(timestamp))
node.put(timestamp_hash, metrics_data)
```

### 3. Load-Balanced Task Distribution
```python
# Distribute tasks across workers
task_hash = node.learned_hash.predict(float(task_id))
worker = node.find_successor(task_hash)
worker.execute_task(task)
```

## Development

### Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v --cov
```

### Code Formatting

```bash
black dlht/
flake8 dlht/
mypy dlht/
```

### Building Documentation

```bash
cd docs
make html
open _build/html/index.html
```

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.


## References


- [A Distributed Learned Hash Table](https://arxiv.org/abs/2508.14239) - Wang et al., 2025
- [The Case for Learned Index Structures](https://arxiv.org/abs/1712.01208) - Kraska et al., 2018
- [Chord: A Scalable Peer-to-peer Lookup Service](https://pdos.csail.mit.edu/papers/chord:sigcomm01/chord_sigcomm.pdf) - Stoica et al., 2001
- [FedAvg: Communication-Efficient Learning](https://arxiv.org/abs/1602.05629) - McMahan et al., 2017

## Support
