"""
Virtual peer (node) implementation for LEAD DHT
"""

import threading
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from sortedcontainers import SortedDict

logger = logging.getLogger(__name__)


@dataclass
class FingerEntry:
    """Entry in virtual finger table"""
    vid: int
    ip: str
    port: int
    
    def to_dict(self) -> dict:
        return {'vid': self.vid, 'ip': self.ip, 'port': self.port}
    
    @classmethod
    def from_dict(cls, data: dict) -> 'FingerEntry':
        return cls(vid=data['vid'], ip=data['ip'], port=data['port'])


@dataclass
class KeyValuePair:
    """Key-value pair with metadata"""
    key: int
    value: Any
    timestamp: float = field(default_factory=time.time)


class LEADPeer:
    """A virtual node/peer in the LEAD overlay"""
    
    def __init__(self, vid: int, ip: str, port: int, physical_node, 
                 hash_space_size: int = 2**160, update_threshold: float = 0.4):
        self.vid = vid
        self.ip = ip
        self.port = port
        self.physical_node = physical_node
        self.hash_space_size = hash_space_size
        self.update_threshold = update_threshold
        
        # Routing state
        self.successor: Optional[FingerEntry] = None
        self.predecessor: Optional[FingerEntry] = None
        self.finger_table: List[Optional[FingerEntry]] = [None] * 160  # log2(2^160)
        self.successor_list: List[FingerEntry] = []  # Backup successors
        self.predecessor_list: List[FingerEntry] = []  # Backup predecessors
        
        # Data storage (using SortedDict for efficient range queries)
        self.storage: SortedDict[int, KeyValuePair] = SortedDict()
        
        # Model update tracking
        self.new_keys_count = 0
        self.total_keys_count = 0
        self.update_ready = False
        
        # Concurrency control
        self.lock = threading.RLock()
        
    def in_range(self, key: int, start: int, end: int, inclusive_end: bool = False) -> bool:
        """Check if key is in range (start, end] on the ring"""
        if start < end:
            return (start < key <= end) if inclusive_end else (start < key < end)
        else:  # Wrap around
            return (key > start or key <= end) if inclusive_end else (key > start or key < end)
            
    def find_successor(self, key_hash: int) -> FingerEntry:
        """Find successor for a given hash value"""
        with self.lock:
            if self.successor is None:
                return FingerEntry(self.vid, self.ip, self.port)
                
            # Check if key is between this node and successor
            if self.in_range(key_hash, self.vid, self.successor.vid, inclusive_end=True):
                return self.successor
                
            # Forward to closest preceding node
            closest = self.closest_preceding_node(key_hash)

            # If closest node is self, return successor (avoid self-RPC)
            if closest.vid == self.vid or \
               (closest.ip == self.ip and closest.port == self.port):
                return self.successor

            # RPC to closest node
            return self.physical_node.rpc_find_successor(closest.ip, closest.port, key_hash)
            
    def closest_preceding_node(self, key_hash: int) -> FingerEntry:
        """Find closest preceding node in finger table"""
        with self.lock:
            # Search finger table in reverse
            for i in range(len(self.finger_table) - 1, -1, -1):
                if self.finger_table[i] is not None:
                    finger = self.finger_table[i]
                    if self.in_range(finger.vid, self.vid, key_hash):
                        return finger
                        
            return FingerEntry(self.vid, self.ip, self.port)
            
    def update_finger_table(self):
        """Update finger table entries"""
        with self.lock:
            for i in range(len(self.finger_table)):
                start = (self.vid + 2**i) % self.hash_space_size
                try:
                    finger = self.find_successor(start)
                    self.finger_table[i] = finger
                except Exception as e:
                    logger.debug(f"Failed to update finger {i}: {e}")
                
    def notify(self, potential_predecessor: FingerEntry):
        """Handle notification of a potential predecessor"""
        with self.lock:
            if self.predecessor is None or \
               self.in_range(potential_predecessor.vid, self.predecessor.vid, self.vid):
                self.predecessor = potential_predecessor
                
    def stabilize(self):
        """Periodic stabilization"""
        with self.lock:
            if self.successor is None:
                return

            # Skip stabilization if successor is self
            if self.successor.vid == self.vid or \
               (self.successor.ip == self.ip and self.successor.port == self.port):
                return

            # Ask successor for its predecessor
            try:
                x = self.physical_node.rpc_get_predecessor(
                    self.successor.ip, self.successor.port)

                if x is not None and self.in_range(x.vid, self.vid, self.successor.vid):
                    self.successor = x

                # Notify successor (skip if successor is now self)
                if self.successor.vid != self.vid and \
                   not (self.successor.ip == self.ip and self.successor.port == self.port):
                    self.physical_node.rpc_notify(
                        self.successor.ip, self.successor.port,
                        FingerEntry(self.vid, self.ip, self.port))

                # Update successor list
                self.update_successor_list()

            except Exception as e:
                logger.debug(f"Stabilization error: {e}")
                self.handle_successor_failure()

    def update_successor_list(self):
        """Update the list of backup successors"""
        max_successors = self.physical_node.config.num_successor_backups
        self.successor_list = []

        try:
            current = self.successor
            for _ in range(max_successors):
                if current.vid == self.vid:
                    break

                # Get successor's successor
                next_succ = self.physical_node.rpc_get_successor(current.ip, current.port)
                if next_succ and next_succ.vid != self.vid and next_succ.vid != current.vid:
                    self.successor_list.append(next_succ)
                    current = next_succ
                else:
                    break

        except Exception as e:
            logger.debug(f"Failed to update successor list: {e}")
                
    def handle_successor_failure(self):
        """Handle successor failure by promoting backup"""
        with self.lock:
            if self.successor_list:
                self.successor = self.successor_list.pop(0)
                logger.info(f"Promoted backup successor: {self.successor.vid}")
            else:
                self.successor = FingerEntry(self.vid, self.ip, self.port)
                
    def put(self, key: int, value: Any):
        """Store a key-value pair"""
        with self.lock:
            is_new = key not in self.storage
            self.storage[key] = KeyValuePair(key, value)

            if is_new:
                self.new_keys_count += 1
                self.total_keys_count += 1

            # Check if we need model update
            # Require minimum number of keys before checking update threshold
            MIN_KEYS_FOR_UPDATE = 15

            if self.total_keys_count >= MIN_KEYS_FOR_UPDATE:
                # Check if new keys exceed threshold relative to existing keys
                baseline_keys = self.total_keys_count - self.new_keys_count

                if baseline_keys > 0:
                    # Have a baseline - check if new keys exceed threshold
                    new_ratio = self.new_keys_count / baseline_keys
                    if new_ratio >= self.update_threshold:
                        self.update_ready = True
                else:
                    # No baseline yet (all keys are new) - use total as baseline
                    # This handles initial accumulation before first model training
                    new_ratio = self.new_keys_count / self.total_keys_count
                    if new_ratio >= self.update_threshold:
                        self.update_ready = True
                    
    def get(self, key: int) -> Optional[Any]:
        """Retrieve a value by key"""
        with self.lock:
            kv = self.storage.get(key)
            return kv.value if kv else None
            
    def get_range(self, start_key: int, count: int) -> List[Tuple[int, Any]]:
        """Get range of keys starting from start_key"""
        with self.lock:
            # SortedDict keeps keys sorted, use efficient range query
            result = []

            # Find index of first key >= start_key using binary search
            idx = self.storage.bisect_left(start_key)

            # Collect count keys starting from idx
            keys = self.storage.keys()
            for i in range(idx, min(idx + count, len(keys))):
                key = keys[i]
                result.append((key, self.storage[key].value))

            return result
            
    def transfer_keys(self, start: int, end: int) -> List[KeyValuePair]:
        """Transfer keys in range to another node"""
        with self.lock:
            keys_to_transfer = []
            keys_to_remove = []
            
            for key, kv in self.storage.items():
                if self.in_range(key, start, end, inclusive_end=True):
                    keys_to_transfer.append(kv)
                    keys_to_remove.append(key)
                    
            for key in keys_to_remove:
                del self.storage[key]
                self.total_keys_count -= 1
                
            return keys_to_transfer
            
    def get_stats(self) -> dict:
        """Get peer statistics"""
        with self.lock:
            return {
                'vid': self.vid,
                'num_keys': len(self.storage),
                'new_keys_count': self.new_keys_count,
                'total_keys_count': self.total_keys_count,
                'update_ready': self.update_ready,
                'has_successor': self.successor is not None,
                'has_predecessor': self.predecessor is not None
            }