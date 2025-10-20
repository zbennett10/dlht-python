"""
Example usage and demonstrations for LEAD DHT Library

This file shows how to use the LEAD library in various scenarios.
"""

import time
import random
import logging
from .node import LEADNode
from .cluster import LEADCluster
from .config import LEADConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_single_node():
    """Example: Simple single-node usage"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Single Node Usage")
    print("="*60 + "\n")
    
    # Create configuration
    config = LEADConfig(
        ip='127.0.0.1',
        base_port=5000,
        num_virtual_nodes=5,
        log_level=logging.INFO
    )
    
    # Create and start node
    print("Creating LEAD node...")
    node = LEADNode(config)
    node.start()
    
    time.sleep(1)  # Allow node to initialize
    
    # Store some data
    print("\nStoring key-value pairs...")
    for i in range(20):
        key = 1000 + i
        value = f"Value_{i}"
        node.put(key, value)
        print(f"  Stored: {key} -> {value}")
    
    # Retrieve data
    print("\nRetrieving some keys...")
    for key in [1005, 1010, 1015]:
        value = node.get(key)
        print(f"  Retrieved: {key} -> {value}")
    
    # Range query
    print("\nExecuting range query (start=1008, count=5)...")
    results = node.range_query(start_key=1008, count=5)
    for key, value in results:
        print(f"  {key} -> {value}")
    
    # Show stats
    print("\nNode Statistics:")
    stats = node.get_stats()
    print(f"  Total Keys: {stats['total_keys']}")
    print(f"  Virtual Nodes: {stats['num_virtual_nodes']}")
    print(f"  Model Version: {stats['model_version']}")
    
    # Cleanup
    print("\nStopping node...")
    node.stop()
    print("Example complete!\n")


def example_multi_node_cluster():
    """Example: Multi-node cluster with data distribution"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Multi-Node Cluster")
    print("="*60 + "\n")
    
    # Create cluster
    cluster = LEADCluster()
    
    # Add first node (bootstrap)
    print("Creating bootstrap node...")
    node1 = cluster.add_node(ip='127.0.0.1', base_port=6000, num_virtual_nodes=5)
    time.sleep(1)
    
    # Add more nodes
    print("Adding additional nodes...")
    node2 = cluster.add_node(
        ip='127.0.0.1', 
        base_port=6100, 
        num_virtual_nodes=5,
        bootstrap_peer=('127.0.0.1', 6000)
    )
    
    node3 = cluster.add_node(
        ip='127.0.0.1',
        base_port=6200,
        num_virtual_nodes=5,
        bootstrap_peer=('127.0.0.1', 6000)
    )
    
    time.sleep(2)  # Allow network to stabilize
    
    # Store data through different nodes
    print("\nStoring 100 key-value pairs across the cluster...")
    for i in range(100):
        key = random.randint(1, 10000)
        value = f"Data_{i}"
        # Randomly use different nodes to insert
        node = random.choice([node1, node2, node3])
        node.put(key, value)
    
    time.sleep(1)
    
    # Query from any node
    print("\nQuerying data from different nodes...")
    print("  Node1 range query:")
    results = node1.range_query(start_key=1000, count=5)
    print(f"    Found {len(results)} keys")
    
    print("  Node2 range query:")
    results = node2.range_query(start_key=5000, count=5)
    print(f"    Found {len(results)} keys")
    
    # Show cluster stats
    cluster.print_stats()
    
    # Cleanup
    print("Stopping cluster...")
    cluster.stop_all()
    print("Example complete!\n")


def example_range_query_performance():
    """Example: Demonstrate range query efficiency"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Range Query Performance")
    print("="*60 + "\n")
    
    cluster = LEADCluster()
    
    # Create cluster
    print("Creating 3-node cluster...")
    node1 = cluster.add_node('127.0.0.1', 7000, num_virtual_nodes=10)
    time.sleep(1)
    
    node2 = cluster.add_node('127.0.0.1', 7100, num_virtual_nodes=10,
                            bootstrap_peer=('127.0.0.1', 7000))
    node3 = cluster.add_node('127.0.0.1', 7200, num_virtual_nodes=10,
                            bootstrap_peer=('127.0.0.1', 7000))
    
    time.sleep(2)
    
    # Insert ordered keys to demonstrate learned hash effectiveness
    print("\nInserting 500 ordered keys...")
    for i in range(500):
        key = i * 100  # Ordered keys with gaps
        value = f"Record_{i}"
        node1.put(key, value)
    
    time.sleep(2)
    
    # Train model on the data
    print("Training learned hash function...")
    node1.retrain_model()
    
    # Perform range queries with timing
    print("\nExecuting range queries with different sizes...")
    
    query_ranges = [10, 25, 50, 100, 200]
    for query_size in query_ranges:
        start_time = time.time()
        results = node1.range_query(10000, query_size)
        elapsed = (time.time() - start_time) * 1000
        
        print(f"  Range size {query_size:3d}: Retrieved {len(results):3d} keys in {elapsed:.2f}ms")
    
    cluster.print_stats()
    
    print("\nStopping cluster...")
    cluster.stop_all()
    print("Example complete!\n")


def example_heterogeneous_cluster():
    """Example: Heterogeneous cluster with different node capacities"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Heterogeneous Cluster")
    print("="*60 + "\n")
    
    cluster = LEADCluster()
    
    print("Creating heterogeneous cluster...")
    print("  - Node 1: 15 virtual nodes (high capacity)")
    print("  - Node 2: 10 virtual nodes (medium capacity)")
    print("  - Node 3: 5 virtual nodes (low capacity)")
    
    node1 = cluster.add_node('127.0.0.1', 8000, num_virtual_nodes=15)
    time.sleep(1)
    
    node2 = cluster.add_node('127.0.0.1', 8100, num_virtual_nodes=10,
                            bootstrap_peer=('127.0.0.1', 8000))
    node3 = cluster.add_node('127.0.0.1', 8200, num_virtual_nodes=5,
                            bootstrap_peer=('127.0.0.1', 8000))
    
    time.sleep(2)
    
    # Insert data
    print("\nInserting 300 keys...")
    for i in range(300):
        key = random.randint(1, 100000)
        value = f"Value_{i}"
        node1.put(key, value)
    
    time.sleep(2)
    
    # Show distribution
    print("\nKey distribution demonstrates load balancing:")
    for idx, node in enumerate(cluster.nodes, 1):
        stats = node.get_stats()
        print(f"  Node {idx}: {stats['num_virtual_nodes']:2d} vnodes, "
              f"{stats['total_keys']:3d} keys, "
              f"avg {stats['avg_keys_per_vnode']:.1f} keys/vnode")
    
    cluster.print_stats()
    
    print("\nStopping cluster...")
    cluster.stop_all()
    print("Example complete!\n")


def example_federated_model_update():
    """Example: Demonstrate federated model updates"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Federated Model Update")
    print("="*60 + "\n")
    
    # Create node with lower update threshold for demo
    config = LEADConfig(
        ip='127.0.0.1',
        base_port=9000,
        num_virtual_nodes=5,
        model_update_threshold=0.3  # Update after 30% new keys
    )
    
    node = LEADNode(config)
    node.start()
    time.sleep(1)
    
    # Initial data
    print("Inserting initial 100 keys...")
    for i in range(100):
        key = i * 10
        value = f"Initial_{i}"
        node.put(key, value)
    
    # Train initial model
    print("Training initial model...")
    node.retrain_model()
    print(f"  Model version: {node.learned_hash.version}")
    
    # Add more keys to trigger update
    print("\nInserting 50 new keys (50% new data)...")
    for i in range(100, 150):
        key = i * 10
        value = f"New_{i}"
        node.put(key, value)
    
    time.sleep(1)
    
    # Check if update is ready
    stats = node.get_stats()
    update_ready_count = sum(1 for v in stats['virtual_nodes'] if v['update_ready'])
    print(f"Virtual nodes ready for update: {update_ready_count}/{stats['num_virtual_nodes']}")
    
    # Perform federated update
    print("\nPerforming federated model update...")
    node.federated_model_update()
    print(f"  Model version: {node.learned_hash.version}")
    
    # Verify update reset counters
    stats = node.get_stats()
    update_ready_count = sum(1 for v in stats['virtual_nodes'] if v['update_ready'])
    print(f"Virtual nodes ready for update after: {update_ready_count}/{stats['num_virtual_nodes']}")
    
    print("\nStopping node...")
    node.stop()
    print("Example complete!\n")


def example_custom_configuration():
    """Example: Using custom configuration"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Custom Configuration")
    print("="*60 + "\n")
    
    # Create custom configuration
    config = LEADConfig(
        ip='127.0.0.1',
        base_port=10000,
        num_virtual_nodes=8,
        model_type='cubic',  # Use cubic model instead of linear
        branching_factor=50,  # Fewer branches
        model_update_threshold=0.5,  # Update at 50% new keys
        rpc_timeout=3.0,
        stabilize_interval=2.0,
        log_level=logging.DEBUG
    )
    
    print("Configuration:")
    print(f"  IP: {config.ip}")
    print(f"  Base Port: {config.base_port}")
    print(f"  Virtual Nodes: {config.num_virtual_nodes}")
    print(f"  Model Type: {config.model_type}")
    print(f"  Branching Factor: {config.branching_factor}")
    print(f"  Update Threshold: {config.model_update_threshold}")
    
    # Create node with custom config
    print("\nCreating node with custom configuration...")
    node = LEADNode(config)
    node.start()
    time.sleep(1)
    
    # Use the node
    print("\nStoring data...")
    for i in range(50):
        node.put(key=i*1000, value=f"Custom_{i}")
    
    stats = node.get_stats()
    print(f"\nStored {stats['total_keys']} keys across {stats['num_virtual_nodes']} virtual nodes")
    
    print("\nStopping node...")
    node.stop()
    print("Example complete!\n")


def example_practical_usage():
    """Example: Practical usage pattern for integration"""
    print("\n" + "="*60)
    print("EXAMPLE 7: Practical Integration Pattern")
    print("="*60 + "\n")
    
    print("This example shows a typical pattern for using LEAD in your application:\n")
    
    # Initialize cluster
    print("1. Initialize the cluster")
    cluster = LEADCluster()
    
    # Add nodes
    print("2. Add nodes to the cluster")
    primary = cluster.add_node('127.0.0.1', 11000, num_virtual_nodes=10)
    time.sleep(1)
    
    # In production, these would be different machines
    cluster.add_node('127.0.0.1', 11100, num_virtual_nodes=10,
                    bootstrap_peer=('127.0.0.1', 11000))
    cluster.add_node('127.0.0.1', 11200, num_virtual_nodes=10,
                    bootstrap_peer=('127.0.0.1', 11000))
    
    time.sleep(2)
    
    # Store application data
    print("3. Store your application data")
    user_data = {
        12345: {"name": "Alice", "email": "alice@example.com"},
        12346: {"name": "Bob", "email": "bob@example.com"},
        12347: {"name": "Charlie", "email": "charlie@example.com"},
        12348: {"name": "Diana", "email": "diana@example.com"},
        12349: {"name": "Eve", "email": "eve@example.com"},
    }
    
    for user_id, data in user_data.items():
        primary.put(user_id, data)
        print(f"   Stored user {user_id}: {data['name']}")
    
    # Retrieve data
    print("\n4. Retrieve data from any node")
    user_id = 12346
    retrieved = primary.get(user_id)
    print(f"   User {user_id}: {retrieved}")
    
    # Range queries for batch operations
    print("\n5. Perform range queries for batch operations")
    results = primary.range_query(start_key=12345, count=3)
    print(f"   Retrieved {len(results)} users in range")
    for uid, data in results:
        print(f"     {uid}: {data['name']}")
    
    # Periodic maintenance
    print("\n6. Periodic maintenance (model updates)")
    print("   Check if update is needed...")
    stats = primary.get_stats()
    if any(v['update_ready'] for v in stats['virtual_nodes']):
        print("   Performing federated update...")
        primary.federated_model_update()
    else:
        print("   No update needed yet")
    
    # Monitor cluster health
    print("\n7. Monitor cluster health")
    cluster.print_stats()
    
    # Graceful shutdown
    print("8. Graceful shutdown")
    cluster.stop_all()
    print("   All nodes stopped cleanly")
    
    print("\nExample complete!\n")


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("LEAD DHT Library - Usage Examples")
    print("="*60)
    
    examples = [
        ("Single Node Usage", example_single_node),
        ("Multi-Node Cluster", example_multi_node_cluster),
        ("Range Query Performance", example_range_query_performance),
        ("Heterogeneous Cluster", example_heterogeneous_cluster),
        ("Federated Model Update", example_federated_model_update),
        ("Custom Configuration", example_custom_configuration),
        ("Practical Integration", example_practical_usage),
    ]
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nRunning all examples...\n")
    
    for name, example_func in examples:
        try:
            example_func()
            time.sleep(1)  # Brief pause between examples
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            break
        except Exception as e:
            logger.error(f"Error in example '{name}': {e}", exc_info=True)
            continue
    
    print("\n" + "="*60)
    print("All examples complete!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()