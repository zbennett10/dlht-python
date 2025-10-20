"""
Configuration module for LEAD DHT
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import logging


@dataclass
class LEADConfig:
    """Configuration for a LEAD node"""
    
    # Network settings
    ip: str = '127.0.0.1'
    base_port: int = 5000
    
    # Node settings
    num_virtual_nodes: int = 10
    bootstrap_peer: Optional[Tuple[str, int]] = None
    
    # Hash space settings
    hash_space_size: int = 2**160  # SHA-1 space
    
    # Model settings
    model_type: str = 'linear'  # 'linear' or 'cubic' for stage 2 (leaf) models
    stage1_model_type: str = 'linear'  # 'linear' or 'cubic' for stage 1 (root) model
    branching_factor: int = 100
    model_update_threshold: float = 0.4  # 40% new keys trigger update
    
    # Network timeouts
    rpc_timeout: float = 5.0  # seconds
    stabilize_interval: float = 1.0  # seconds
    
    # Thread pool settings
    max_workers: int = 20
    
    # Backup/redundancy settings
    num_successor_backups: int = 3
    num_predecessor_backups: int = 3
    
    # Logging
    log_level: int = logging.DEBUG  # Enable DEBUG for troubleshooting
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    def __post_init__(self):
        """Validate configuration"""
        if self.num_virtual_nodes < 1:
            raise ValueError("num_virtual_nodes must be at least 1")

        if self.model_type not in ('linear', 'cubic'):
            raise ValueError("model_type must be 'linear' or 'cubic'")

        if self.stage1_model_type not in ('linear', 'cubic'):
            raise ValueError("stage1_model_type must be 'linear' or 'cubic'")

        if self.branching_factor < 10:
            raise ValueError("branching_factor must be at least 10")

        if not 0 < self.model_update_threshold <= 1.0:
            raise ValueError("model_update_threshold must be between 0 and 1")
            
        # Configure logging
        logging.basicConfig(
            level=self.log_level,
            format=self.log_format
        )
        
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            'ip': self.ip,
            'base_port': self.base_port,
            'num_virtual_nodes': self.num_virtual_nodes,
            'bootstrap_peer': self.bootstrap_peer,
            'hash_space_size': self.hash_space_size,
            'model_type': self.model_type,
            'branching_factor': self.branching_factor,
            'model_update_threshold': self.model_update_threshold,
            'rpc_timeout': self.rpc_timeout,
            'stabilize_interval': self.stabilize_interval,
            'max_workers': self.max_workers
        }
        
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'LEADConfig':
        """Create config from dictionary"""
        return cls(**config_dict)