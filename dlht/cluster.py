"""
Cluster management for multiple LEAD nodes
"""

import time
import logging
from typing import List, Optional, Tuple

from .node import LEADNode
from .config import LEADConfig

logger = logging.getLogger(__name__)


class LEADCluster:
    """Manages a cluster of LEAD nodes"""
    
    def __init__(self):
        self.nodes: List[LEADNode] = []
        
    def add_node(self, ip: str = '127.0.0.1', base_port: int = 5000, 
                 num_virtual_nodes: int = 10,
                 bootstrap_peer: Optional[Tuple[str, int]] = None,
                 config: Optional[LEADConfig] = None) -> LEADNode:
        """
        Add a new node to the cluster
        
        Args:
            ip: IP address for the node
            base_port: Base port for virtual nodes
            num_virtual_nodes: Number of virtual nodes per physical node
            bootstrap_peer: (ip, port) tuple of existing node to join
            config: Optional LEADConfig object (overrides other params)
            
        Returns:
            The created LEADNode instance
        """
        if config is None:
            config = LEADConfig(
                ip=ip,
                base_port=base_port,
                num_virtual_nodes=num_virtual_nodes,
                bootstrap_peer=bootstrap_peer
            )
        
        node = LEADNode(config)
        node.start()
        self.nodes.append(node)
        
        # Allow time for node to stabilize
        time.sleep(0.5)
        
        logger.info(f"Added node {ip}:{base_port} to cluster")
        return node
        
    def stop_all(self):
        """Stop all nodes in the cluster"""
        logger.info("Stopping all nodes in cluster")
        for node in self.nodes:
            node.stop()
        self.nodes.clear()
            
    def get_stats(self) -> dict:
        """
        Get cluster-wide statistics
        
        Returns:
            Dictionary with cluster statistics
        """
        total_keys = 0
        total_vnodes = 0
        node_stats = []
        
        for node in self.nodes:
            stats = node.get_stats()
            node_stats.append(stats)
            total_keys += stats['total_keys']
            total_vnodes += stats['num_virtual_nodes']
            
        return {
            'num_physical_nodes': len(self.nodes),
            'num_virtual_nodes': total_vnodes,
            'total_keys': total_keys,
            'avg_keys_per_vnode': total_keys / total_vnodes if total_vnodes > 0 else 0,
            'avg_keys_per_pnode': total_keys / len(self.nodes) if self.nodes else 0,
            'nodes': node_stats
        }
        
    def print_stats(self):
        """Print cluster statistics to console"""
        stats = self.get_stats()
        print("\n" + "="*60)
        print("LEAD CLUSTER STATISTICS")
        print("="*60)
        print(f"Physical Nodes:     {stats['num_physical_nodes']}")
        print(f"Virtual Nodes:      {stats['num_virtual_nodes']}")
        print(f"Total Keys:         {stats['total_keys']}")
        print(f"Avg Keys/VNode:     {stats['avg_keys_per_vnode']:.2f}")
        print(f"Avg Keys/PNode:     {stats['avg_keys_per_pnode']:.2f}")
        print("="*60)
        
        for i, node_stat in enumerate(stats['nodes'], 1):
            print(f"\nNode {i} ({node_stat['ip']}:{node_stat['base_port']}):")
            print(f"  Virtual Nodes:    {node_stat['num_virtual_nodes']}")
            print(f"  Keys:             {node_stat['total_keys']}")
            print(f"  Model Version:    {node_stat['model_version']}")
            print(f"  Status:           {'Ready' if node_stat['ready'] else 'Not Ready'}")
        
        print("="*60 + "\n")
        
    def get_node(self, index: int) -> Optional[LEADNode]:
        """
        Get node by index
        
        Args:
            index: Node index (0-based)
            
        Returns:
            LEADNode instance or None if index out of range
        """
        if 0 <= index < len(self.nodes):
            return self.nodes[index]
        return None
        
    def __len__(self) -> int:
        """Return number of nodes in cluster"""
        return len(self.nodes)
        
    def __getitem__(self, index: int) -> LEADNode:
        """Allow indexing to get nodes"""
        return self.nodes[index]
        
    def __iter__(self):
        """Allow iteration over nodes"""
        return iter(self.nodes)