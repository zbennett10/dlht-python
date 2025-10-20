"""
Physical node implementation for LEAD DHT
"""

import socket
import struct
import threading
import time
import logging
import hashlib
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

from .config import LEADConfig
from .models import RecursiveModelIndex
from .adaptive_training import AdaptiveTrainingManager, TrainingConfig, federated_average_models
from .peer import LEADPeer, FingerEntry, KeyValuePair
from .utils import peer_hash, distance, RPCMessage
from .exceptions import NetworkException, RPCException, NodeNotReadyException

logger = logging.getLogger(__name__)


class LEADNode:
    """Physical node that hosts multiple virtual nodes"""
    
    def __init__(self, config: Optional[LEADConfig] = None):
        """
        Initialize a LEAD node
        
        Args:
            config: LEADConfig object with node configuration
        """
        self.config = config or LEADConfig()
        
        self.ip = self.config.ip
        self.base_port = self.config.base_port
        self.num_virtual_nodes = self.config.num_virtual_nodes
        self.bootstrap_peer = self.config.bootstrap_peer
        self.hash_space_size = self.config.hash_space_size
        
        # Virtual nodes
        self.virtual_nodes: Dict[int, LEADPeer] = {}
        
        # Learned hash function (shared across virtual nodes)
        self.learned_hash = RecursiveModelIndex(
            branching_factor=self.config.branching_factor,
            model_type=self.config.model_type,
            hash_space_size=self.hash_space_size,
            stage1_model_type=self.config.stage1_model_type
        )
        self.learned_hash_lock = threading.RLock()

        # Adaptive training manager for intelligent retraining
        self.training_manager = AdaptiveTrainingManager(TrainingConfig(
            enable_error_trigger=True,
            enable_volume_trigger=True,
            enable_staleness_trigger=True,
            max_staleness_hours=24.0,
            new_keys_trigger_count=1000,
            new_keys_trigger_percent=0.1,
            reservoir_sample_size=10000
        ))

        # Routing table: Maps tip_hash -> list of entities (CDC-populated)
        # Multiple entities can have the same hash, so we store a list per hash
        # This is the lightweight metadata store for routing queries
        self.routing_table: Dict[int, List[Dict[str, Any]]] = {}
        self.routing_table_lock = threading.RLock()

        # Network
        self.server_socket = None
        self.running = False
        self.ready = False
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        # Stabilization
        self.stabilize_interval = self.config.stabilize_interval
        
    def start(self):
        """Start the node and join the network"""
        if self.running:
            logger.warning("Node already running")
            return
            
        logger.info(f"Starting LEAD node at {self.ip}:{self.base_port}")
        
        # Create virtual nodes
        # All virtual nodes share the same port (base_port) since they're logical entities
        # on the same physical node. VID is generated using a salt to distribute them in hash space.
        for i in range(self.num_virtual_nodes):
            # Generate unique VID for each virtual node using salt
            vid_data = f"{self.ip}:{self.base_port}:vnode{i}".encode()
            vid = int.from_bytes(hashlib.sha1(vid_data).digest(), 'big')

            vnode = LEADPeer(
                vid, self.ip, self.base_port, self,
                self.hash_space_size,
                self.config.model_update_threshold
            )
            self.virtual_nodes[vid] = vnode
            
        logger.info(f"Created {len(self.virtual_nodes)} virtual nodes")
        
        # Initialize learned hash with empty training
        self.learned_hash.train(np.array([]))
        
        # Start network server
        self.running = True
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()

        # Give server time to start listening (especially important in Docker)
        time.sleep(0.5)

        # Join network
        if self.bootstrap_peer:
            self._join_network()
        else:
            # First node in network
            for vnode in self.virtual_nodes.values():
                vnode.successor = FingerEntry(vnode.vid, vnode.ip, vnode.port)
                vnode.predecessor = FingerEntry(vnode.vid, vnode.ip, vnode.port)
            logger.info("Initialized as first node in network")
                
        # Start stabilization
        self.stabilize_thread = threading.Thread(target=self._stabilize_loop, daemon=True)
        self.stabilize_thread.start()
        
        self.ready = True
        logger.info(f"Node started successfully")
        
    def stop(self):
        """Stop the node"""
        if not self.running:
            return
            
        logger.info("Stopping LEAD node")
        self.running = False
        self.ready = False
        
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
                
        self.executor.shutdown(wait=False)
        logger.info("Node stopped")
        
    def _join_network(self):
        """Join existing network via bootstrap peer"""
        logger.info(f"Joining network via bootstrap peer {self.bootstrap_peer}")

        # Retry joining if bootstrap peer isn't ready yet
        max_retries = 5
        retry_delay = 2  # seconds

        for vnode in self.virtual_nodes.values():
            for attempt in range(max_retries):
                try:
                    # Find successor via bootstrap
                    successor = self.rpc_find_successor(
                        self.bootstrap_peer[0], self.bootstrap_peer[1], vnode.vid)

                    vnode.successor = successor
                    vnode.predecessor = None

                    # Update finger table
                    vnode.update_finger_table()

                    # Notify successor
                    self.rpc_notify(successor.ip, successor.port,
                                   FingerEntry(vnode.vid, vnode.ip, vnode.port))

                    # Request key transfer
                    self._rpc_request_keys(successor.ip, successor.port,
                                          successor.vid, vnode.vid)

                    logger.info(f"Virtual node {vnode.vid} successfully joined network")
                    break  # Success, exit retry loop

                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Join attempt {attempt + 1}/{max_retries} failed for vnode {vnode.vid}: {e}. Retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"Failed to join network for vnode {vnode.vid} after {max_retries} attempts: {e}")
                
    def _run_server(self):
        """Run RPC server"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # Bind to 0.0.0.0 to accept connections from all interfaces (required for Docker)
            bind_ip = '0.0.0.0'
            self.server_socket.bind((bind_ip, self.base_port))
            self.server_socket.listen(100)

            logger.info(f"RPC server listening on {bind_ip}:{self.base_port} (advertised as {self.ip}:{self.base_port})")
            
            while self.running:
                try:
                    conn, addr = self.server_socket.accept()
                    self.executor.submit(self._handle_connection, conn)
                except Exception as e:
                    if self.running:
                        logger.error(f"Server accept error: {e}")
                    break
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            self.running = False
                
    def _handle_connection(self, conn: socket.socket):
        """Handle incoming RPC connection"""
        try:
            logger.debug(f"Handling incoming RPC connection")
            # Read header
            header = conn.recv(4)
            logger.debug(f"Received header: {len(header)} bytes")
            if len(header) < 4:
                logger.warning("Incomplete header received")
                return

            msg_len = struct.unpack('!I', header)[0]
            logger.debug(f"Expecting message of {msg_len} bytes")

            # Read message
            data = b''
            while len(data) < msg_len:
                chunk = conn.recv(min(msg_len - len(data), 4096))
                if not chunk:
                    break
                data += chunk
            logger.debug(f"Received {len(data)} bytes of message data")

            msg_type, payload = RPCMessage.decode(data)
            logger.debug(f"Decoded RPC request: {msg_type}")

            # Route to handler
            response = self._handle_rpc(msg_type, payload)
            logger.debug(f"Generated response for {msg_type}")

            # Send response
            response_data = RPCMessage.encode('response', response)
            logger.debug(f"Sending response: {len(response_data)} bytes")
            conn.sendall(response_data)
            logger.debug(f"Response sent successfully for {msg_type}")

        except Exception as e:
            logger.error(f"Error handling connection: {e}", exc_info=True)
        finally:
            conn.close()
            
    def _handle_rpc(self, msg_type: str, payload: dict) -> dict:
        """Handle RPC request"""
        try:
            if msg_type == 'find_successor':
                key_hash = payload['key_hash']
                vnode = self._get_vnode_for_hash(key_hash)
                successor = vnode.find_successor(key_hash)
                
                return {
                    'success': True,
                    'successor': successor.to_dict()
                }
                
            elif msg_type == 'get_predecessor':
                vnode_vid = payload.get('vnode_vid')
                if vnode_vid and vnode_vid in self.virtual_nodes:
                    vnode = self.virtual_nodes[vnode_vid]
                else:
                    vnode = list(self.virtual_nodes.values())[0]

                if vnode.predecessor:
                    return {
                        'success': True,
                        'predecessor': vnode.predecessor.to_dict()
                    }
                return {'success': False}

            elif msg_type == 'get_successor':
                vnode_vid = payload.get('vnode_vid')
                if vnode_vid and vnode_vid in self.virtual_nodes:
                    vnode = self.virtual_nodes[vnode_vid]
                else:
                    vnode = list(self.virtual_nodes.values())[0]

                if vnode.successor:
                    return {
                        'success': True,
                        'successor': vnode.successor.to_dict()
                    }
                return {'success': False}
                
            elif msg_type == 'notify':
                vnode_vid = payload.get('vnode_vid')
                potential_pred = FingerEntry.from_dict(payload['predecessor'])
                
                if vnode_vid and vnode_vid in self.virtual_nodes:
                    vnode = self.virtual_nodes[vnode_vid]
                    vnode.notify(potential_pred)
                else:
                    # Notify all virtual nodes
                    for vnode in self.virtual_nodes.values():
                        vnode.notify(potential_pred)
                    
                return {'success': True}
                
            elif msg_type == 'put':
                key = payload['key']
                value = payload['value']
                
                vnode = self._get_vnode_for_hash(key)
                vnode.put(key, value)
                
                return {'success': True}
                
            elif msg_type == 'get':
                key = payload['key']
                vnode = self._get_vnode_for_hash(key)
                value = vnode.get(key)
                
                return {'success': True, 'value': value}
                
            elif msg_type == 'get_range':
                start_key = payload['start_key']
                count = payload['count']
                
                vnode = self._get_vnode_for_hash(start_key)
                results = vnode.get_range(start_key, count)
                
                return {'success': True, 'results': results}
                
            elif msg_type == 'request_keys':
                vnode_vid = payload['vnode_vid']
                new_node_vid = payload['new_node_vid']
                
                if vnode_vid in self.virtual_nodes:
                    vnode = self.virtual_nodes[vnode_vid]
                    pred_vid = vnode.predecessor.vid if vnode.predecessor else 0
                    keys = vnode.transfer_keys(pred_vid, new_node_vid)
                    return {
                        'success': True,
                        'keys': [(kv.key, kv.value) for kv in keys]
                    }
                    
                return {'success': False}
                
            else:
                return {'success': False, 'error': 'Unknown message type'}
                
        except Exception as e:
            logger.error(f"Error handling RPC {msg_type}: {e}")
            return {'success': False, 'error': str(e)}
            
    def _get_vnode_for_hash(self, key_hash: int) -> LEADPeer:
        """Get the virtual node responsible for a hash"""
        # Find closest virtual node by VID
        closest_vnode = None
        min_distance = float('inf')
        
        for vnode in self.virtual_nodes.values():
            dist = distance(key_hash, vnode.vid, self.hash_space_size)
            if dist < min_distance:
                min_distance = dist
                closest_vnode = vnode
                
        return closest_vnode
        
    def _stabilize_loop(self):
        """Periodic stabilization loop"""
        while self.running:
            try:
                for vnode in self.virtual_nodes.values():
                    vnode.stabilize()
                    vnode.update_finger_table()
                    
                time.sleep(self.stabilize_interval)
            except Exception as e:
                logger.error(f"Stabilization error: {e}")
                
    # ========================================================================
    # PUBLIC API
    # ========================================================================
    
    def put(self, key: int, value: Any):
        """
        Store key-value pair in the DHT
        
        Args:
            key: Integer key
            value: Value to store (any serializable object)
        """
        if not self.ready:
            raise NodeNotReadyException("Node is not ready")
            
        # Hash key using learned hash function
        with self.learned_hash_lock:
            hash_value = self.learned_hash.predict(key)
            
        # Find responsible node
        vnode = self._get_vnode_for_hash(hash_value)
        successor = vnode.find_successor(hash_value)

        # Store locally or forward
        if successor.ip == self.ip and successor.port == self.base_port:
            # Successor is a local vnode
            if successor.vid in self.virtual_nodes:
                self.virtual_nodes[successor.vid].put(key, value)
            else:
                # Shouldn't happen, but fallback to original vnode
                vnode.put(key, value)
        else:
            # Successor is on a different node
            self.rpc_put(successor.ip, successor.port, key, value)
            
    def get(self, key: int) -> Optional[Any]:
        """
        Retrieve value by key

        Args:
            key: Integer key

        Returns:
            Value associated with key, or None if not found
        """
        if not self.ready:
            raise NodeNotReadyException("Node is not ready")

        with self.learned_hash_lock:
            hash_value = self.learned_hash.predict(key)

        logger.debug(f"get({key}): predicted hash={hash_value}")

        vnode = self._get_vnode_for_hash(hash_value)
        successor = vnode.find_successor(hash_value)

        logger.debug(f"get({key}): vnode={vnode.vid}, successor={successor.vid} at {successor.ip}:{successor.port}")

        # Check if successor is a local vnode (same IP and port as this node)
        if successor.ip == self.ip and successor.port == self.base_port:
            logger.debug(f"get({key}): successor is local, checking local vnodes")

            # Try to get from the specific successor vnode first
            if successor.vid in self.virtual_nodes:
                result = self.virtual_nodes[successor.vid].get(key)
                logger.debug(f"get({key}): local vnode {successor.vid} returned {result}")
                if result is not None:
                    return result

            # Fallback: check all local vnodes (handles case where model was retrained)
            logger.debug(f"get({key}): checking all {len(self.virtual_nodes)} local vnodes")
            for vid, local_vnode in self.virtual_nodes.items():
                value = local_vnode.get(key)
                logger.debug(f"get({key}): checking vnode {vid}, got {value}")
                if value is not None:
                    logger.info(f"get({key}): found in local vnode {vid}")
                    return value

            logger.warning(f"get({key}): not found in any local vnode")
            return None
        else:
            # Successor is on a different node - use RPC
            logger.debug(f"get({key}): trying RPC to remote successor {successor.ip}:{successor.port}")
            try:
                result = self.rpc_get(successor.ip, successor.port, key)
                logger.debug(f"get({key}): RPC returned {result}")
                return result
            except (RPCException, Exception) as e:
                logger.error(f"get({key}): RPC failed with {e}")
                return None
            
    def range_query(self, start_key: int, count: int) -> List[Tuple[int, Any]]:
        """
        Execute range query for sequential keys
        
        Args:
            start_key: Starting key for range
            count: Number of keys to retrieve
            
        Returns:
            List of (key, value) tuples
        """
        if not self.ready:
            raise NodeNotReadyException("Node is not ready")
            
        with self.learned_hash_lock:
            hash_value = self.learned_hash.predict(start_key)
            
        vnode = self._get_vnode_for_hash(hash_value)
        successor = vnode.find_successor(hash_value)
        
        results = []
        current_node = successor
        remaining = count
        
        # Limit iterations to prevent infinite loops
        max_iterations = self.num_virtual_nodes * 10
        iteration = 0
        
        while remaining > 0 and len(results) < count and iteration < max_iterations:
            iteration += 1
            
            # Get range from current node
            try:
                if current_node.vid == vnode.vid:
                    batch = vnode.get_range(start_key, remaining)
                else:
                    batch = self.rpc_get_range(current_node.ip, current_node.port, 
                                              start_key, remaining)
                    
                if not batch:
                    break
                    
                results.extend(batch)
                remaining -= len(batch)
                
                # Move to successor if we need more keys
                if remaining > 0:
                    if current_node.vid == vnode.vid:
                        if vnode.successor and vnode.successor.vid != vnode.vid:
                            current_node = vnode.successor
                        else:
                            break
                    else:
                        # Get successor of remote node
                        succ = self.rpc_find_successor(
                            current_node.ip, current_node.port,
                            (current_node.vid + 1) % self.hash_space_size
                        )
                        if succ.vid == current_node.vid:
                            break
                        current_node = succ
                        
                    # Update start_key for next batch
                    if results:
                        start_key = results[-1][0] + 1
            except Exception as e:
                logger.error(f"Error in range query: {e}")
                break
                    
    def retrain_model(self, sample_keys: Optional[np.ndarray] = None, force: bool = False):
        """
        Retrain the learned hash function with adaptive strategy

        Args:
            sample_keys: Optional array of keys to train on.
                        If None, uses reservoir sample or collects from routing table
            force: Force retraining regardless of adaptive triggers
        """
        # Check if we should retrain (unless forced)
        should_retrain, reason = self.training_manager.should_retrain(force=force)

        if not should_retrain and not force:
            logger.debug(f"Skipping retrain: {reason}")
            return

        logger.info(f"Starting model retraining: {reason}")

        with self.learned_hash_lock:
            if sample_keys is None:
                # Try to use reservoir sample first (more efficient)
                # Fall back to collecting from routing table if reservoir is empty
                with self.routing_table_lock:
                    all_keys = list(self.routing_table.keys())

                if all_keys:
                    all_keys_array = np.array(sorted([float(k) for k in all_keys]))
                    sample_keys = self.training_manager.get_training_data(all_keys_array)
                else:
                    logger.warning("No keys available for training")
                    return

            if len(sample_keys) == 0:
                logger.warning("No keys to train model on")
                return

            # Train the model
            self.learned_hash.train(sample_keys)

            # Notify training manager
            self.training_manager.after_training(len(sample_keys), self.learned_hash.version)

            # Reset new key counters (if virtual nodes exist)
            for vnode in self.virtual_nodes.values():
                with vnode.lock:
                    vnode.new_keys_count = 0
                    vnode.update_ready = False

            logger.info(f"✓ Model retrained: version {self.learned_hash.version}, "
                       f"{len(sample_keys)} keys, reason: {reason}") 
    # def retrain_model(self, sample_keys: Optional[np.ndarray] = None):
    #     """
    #     Retrain the learned hash function
        
    #     Args:
    #         sample_keys: Optional array of keys to train on. 
    #                     If None, collects keys from all virtual nodes
    #     """
    #     with self.learned_hash_lock:
    #         if sample_keys is None:
    #             # Collect keys from all virtual nodes
    #             all_keys = []
    #             for vnode in self.virtual_nodes.values():
    #                 with vnode.lock:
    #                     all_keys.extend(vnode.storage.keys())
                        
    #             if not all_keys:
    #                 logger.warning("No keys to train model on")
    #                 return
                    
    #             sample_keys = np.array(sorted(all_keys))
                
    #         self.learned_hash.train(sample_keys)
            
    #         # Reset new key counters
    #         for vnode in self.virtual_nodes.values():
    #             with vnode.lock:
    #                 vnode.new_keys_count = 0
    #                 vnode.update_ready = False
                    
    #         logger.info(f"Model retrained with {len(sample_keys)} keys, "
    #                    f"version {self.learned_hash.version}")
            
    def federated_model_update(self):
        """Perform federated model update across virtual nodes"""
        # Check if majority of virtual nodes are ready for update
        ready_count = sum(1 for vnode in self.virtual_nodes.values() 
                         if vnode.update_ready)
        
        if ready_count < len(self.virtual_nodes) * 0.9:
            logger.debug(f"Not enough nodes ready for update ({ready_count}/{len(self.virtual_nodes)})")
            return
            
        logger.info("Initiating federated model update")
        
        # Collect leaf model updates from local virtual nodes
        leaf_updates = defaultdict(list)
        
        for vnode in self.virtual_nodes.values():
            with vnode.lock:
                if not vnode.storage:
                    continue
                    
                # Get keys and their relative positions
                sorted_keys = sorted(vnode.storage.keys())
                
                for idx, key in enumerate(sorted_keys):
                    # Predict which leaf model this key belongs to
                    normalized_key = key / self.hash_space_size
                    bucket_prediction = self.learned_hash.stage1_model.predict(normalized_key)
                    bucket_id = int(np.clip(bucket_prediction, 0, 
                                           self.learned_hash.branching_factor - 1))
                    
                    # Calculate relative position for training
                    relative_pos = idx / len(sorted_keys)
                    leaf_updates[bucket_id].append((key, relative_pos))
                    
        # Update leaf models with aggregated data
        with self.learned_hash_lock:
            for bucket_id, key_pos_pairs in leaf_updates.items():
                if len(key_pos_pairs) > 2:  # Need minimum data points
                    keys = np.array([kp[0] for kp in key_pos_pairs])
                    positions = np.array([kp[1] for kp in key_pos_pairs])
                    self.learned_hash.update_leaf(bucket_id, keys, positions)
                    
            self.learned_hash.version += 1
            
        # Reset update flags
        for vnode in self.virtual_nodes.values():
            with vnode.lock:
                vnode.new_keys_count = 0
                vnode.update_ready = False
                
        logger.info(f"Federated model update complete, version {self.learned_hash.version}")
        
    def get_stats(self) -> dict:
        """
        Get node statistics
        
        Returns:
            Dictionary with node statistics
        """
        total_keys = 0
        vnode_stats = []
        
        for vnode in self.virtual_nodes.values():
            stats = vnode.get_stats()
            vnode_stats.append(stats)
            total_keys += stats['num_keys']
            
        return {
            'ip': self.ip,
            'base_port': self.base_port,
            'num_virtual_nodes': len(self.virtual_nodes),
            'total_keys': total_keys,
            'avg_keys_per_vnode': total_keys / len(self.virtual_nodes) if self.virtual_nodes else 0,
            'model_version': self.learned_hash.version,
            'running': self.running,
            'ready': self.ready,
            'virtual_nodes': vnode_stats
        }
        
    # ========================================================================
    # RPC CLIENT METHODS
    # ========================================================================
    
    def _send_rpc(self, ip: str, port: int, msg_type: str, payload: dict) -> dict:
        """Send RPC request and get response"""
        try:
            logger.debug(f"Sending RPC {msg_type} to {ip}:{port}")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.config.rpc_timeout)

            logger.debug(f"Connecting to {ip}:{port}...")
            sock.connect((ip, port))
            logger.debug(f"Connected to {ip}:{port}")

            # Send request
            request = RPCMessage.encode(msg_type, payload)
            logger.debug(f"Sending request of {len(request)} bytes")
            sock.sendall(request)
            logger.debug(f"Request sent, waiting for response...")

            # Receive response
            header = sock.recv(4)
            logger.debug(f"Received header: {len(header)} bytes")
            if len(header) < 4:
                return {'success': False, 'error': 'Invalid response'}

            msg_len = struct.unpack('!I', header)[0]
            logger.debug(f"Expecting response of {msg_len} bytes")

            data = b''
            while len(data) < msg_len:
                chunk = sock.recv(min(msg_len - len(data), 4096))
                if not chunk:
                    break
                data += chunk
            logger.debug(f"Received {len(data)} bytes of data")

            _, response = RPCMessage.decode(data)
            sock.close()
            logger.debug(f"RPC {msg_type} completed successfully")

            return response

        except socket.timeout:
            logger.error(f"RPC timeout to {ip}:{port} for {msg_type}")
            raise RPCException(f"RPC timeout to {ip}:{port}")
        except Exception as e:
            logger.error(f"RPC error to {ip}:{port} for {msg_type}: {e}")
            raise RPCException(f"RPC error to {ip}:{port}: {e}")
            
    def rpc_find_successor(self, ip: str, port: int, key_hash: int) -> FingerEntry:
        """RPC: Find successor of a hash value"""
        response = self._send_rpc(ip, port, 'find_successor', 
                                 {'key_hash': key_hash})
                                 
        if response.get('success'):
            return FingerEntry.from_dict(response['successor'])
        else:
            raise RPCException(f"Failed to find successor: {response.get('error')}")
            
    def rpc_get_predecessor(self, ip: str, port: int) -> Optional[FingerEntry]:
        """RPC: Get predecessor of a node"""
        response = self._send_rpc(ip, port, 'get_predecessor', {})

        if response.get('success') and 'predecessor' in response:
            return FingerEntry.from_dict(response['predecessor'])
        return None

    def rpc_get_successor(self, ip: str, port: int) -> Optional[FingerEntry]:
        """RPC: Get successor of a node"""
        response = self._send_rpc(ip, port, 'get_successor', {})

        if response.get('success') and 'successor' in response:
            return FingerEntry.from_dict(response['successor'])
        return None
        
    def rpc_notify(self, ip: str, port: int, potential_pred: FingerEntry):
        """RPC: Notify node of potential predecessor"""
        payload = {'predecessor': potential_pred.to_dict()}
        self._send_rpc(ip, port, 'notify', payload)
        
    def rpc_put(self, ip: str, port: int, key: int, value: Any):
        """RPC: Store key-value pair"""
        response = self._send_rpc(ip, port, 'put', {'key': key, 'value': value})
        if not response.get('success'):
            raise RPCException(f"Failed to put key: {response.get('error')}")
            
    def rpc_get(self, ip: str, port: int, key: int) -> Optional[Any]:
        """RPC: Retrieve value by key"""
        response = self._send_rpc(ip, port, 'get', {'key': key})
        if response.get('success'):
            return response.get('value')
        return None
        
    def rpc_get_range(self, ip: str, port: int, start_key: int, 
                     count: int) -> List[Tuple[int, Any]]:
        """RPC: Get range of keys"""
        response = self._send_rpc(ip, port, 'get_range', 
                                 {'start_key': start_key, 'count': count})
        if response.get('success'):
            return response.get('results', [])
        return []
        
    def _rpc_request_keys(self, ip: str, port: int, vnode_vid: int, new_node_vid: int):
        """RPC: Request key transfer from successor"""
        response = self._send_rpc(ip, port, 'request_keys',
                                 {'vnode_vid': vnode_vid, 'new_node_vid': new_node_vid})
        if response.get('success'):
            keys = response.get('keys', [])
            # Store transferred keys locally
            for key, value in keys:
                self.put(key, value)

    def get_nodes_for_tip_range(self, hash_min: int, hash_max: int) -> List[Dict[str, Any]]:
        """
        Find all nodes (virtual and physical) that might contain data
        in the given hash range.

        This is the key method for TIP range query routing - it determines
        which registry nodes should be queried for a given TIP range.

        Args:
            hash_min: Minimum hash value (from TIP range hashing)
            hash_max: Maximum hash value (from TIP range hashing)

        Returns:
            List of node info dicts with keys:
            - vid: Virtual node ID
            - ip: Physical node IP
            - port: Physical node port
            - hash_range: (start, end) range this vnode is responsible for
            - is_local: Whether this is a local vnode
        """
        if not self.ready:
            raise NodeNotReadyException("Node is not ready")

        nodes = []
        seen_physical_nodes = set()  # Track unique physical nodes

        # Find all virtual nodes whose ranges overlap [hash_min, hash_max]
        # We need to walk the ring from hash_min to hash_max

        with self.learned_hash_lock:
            # Start with the vnode responsible for hash_min
            start_vnode = self._get_vnode_for_hash(hash_min)
            start_successor = start_vnode.find_successor(hash_min)

            current_hash = hash_min
            current_successor = start_successor
            max_iterations = self.num_virtual_nodes * 100  # Prevent infinite loops
            iteration = 0

            while current_hash <= hash_max and iteration < max_iterations:
                iteration += 1

                # Get range for current successor
                if current_successor.vid in self.virtual_nodes:
                    # Local vnode
                    local_vnode = self.virtual_nodes[current_successor.vid]
                    pred = local_vnode.predecessor

                    # Calculate range this vnode is responsible for
                    if pred:
                        range_start = (pred.vid + 1) % self.hash_space_size
                    else:
                        range_start = current_successor.vid

                    range_end = current_successor.vid

                    # Check if this range overlaps with [hash_min, hash_max]
                    if self._ranges_overlap(range_start, range_end, hash_min, hash_max):
                        physical_key = f"{current_successor.ip}:{current_successor.port}"

                        nodes.append({
                            'vid': current_successor.vid,
                            'ip': current_successor.ip,
                            'port': current_successor.port,
                            'hash_range': (range_start, range_end),
                            'is_local': True,
                            'physical_node': physical_key
                        })
                        seen_physical_nodes.add(physical_key)

                else:
                    # Remote vnode - query via RPC
                    try:
                        # Get predecessor to calculate range
                        pred = self.rpc_get_predecessor(current_successor.ip, current_successor.port)

                        if pred:
                            range_start = (pred.vid + 1) % self.hash_space_size
                        else:
                            range_start = current_successor.vid

                        range_end = current_successor.vid

                        if self._ranges_overlap(range_start, range_end, hash_min, hash_max):
                            physical_key = f"{current_successor.ip}:{current_successor.port}"

                            nodes.append({
                                'vid': current_successor.vid,
                                'ip': current_successor.ip,
                                'port': current_successor.port,
                                'hash_range': (range_start, range_end),
                                'is_local': False,
                                'physical_node': physical_key
                            })
                            seen_physical_nodes.add(physical_key)

                    except Exception as e:
                        logger.warning(f"Failed to query remote node {current_successor.ip}:{current_successor.port}: {e}")

                # Move to next successor in ring
                try:
                    if current_successor.vid in self.virtual_nodes:
                        next_succ = self.virtual_nodes[current_successor.vid].successor
                    else:
                        next_succ = self.rpc_get_successor(current_successor.ip, current_successor.port)

                    if not next_succ or next_succ.vid == current_successor.vid:
                        # Reached end of ring
                        break

                    current_successor = next_succ
                    current_hash = current_successor.vid

                except Exception as e:
                    logger.warning(f"Failed to get successor: {e}")
                    break

        logger.info(f"Found {len(nodes)} vnodes ({len(seen_physical_nodes)} unique physical nodes) for hash range [{hash_min}, {hash_max}]")
        return nodes

    def _ranges_overlap(self, range_start: int, range_end: int,
                       query_min: int, query_max: int) -> bool:
        """
        Check if a vnode's hash range overlaps with query range.

        Handles wraparound in circular hash space.
        """
        # Normalize wraparound case
        if range_start > range_end:
            # Range wraps around: [range_start, MAX] and [0, range_end]
            return (query_min <= range_end or query_max >= range_start)
        else:
            # Normal case: [range_start, range_end]
            return not (query_max < range_start or query_min > range_end)

    def query_tip_range(self, hash_min: int, hash_max: int) -> Dict[str, Any]:
        """
        Execute a TIP range query and return routing information.

        This returns information about which nodes to query, but does NOT
        execute the actual PostgreSQL queries (that's done at a higher level).

        Args:
            hash_min: Minimum hash value from TIP range
            hash_max: Maximum hash value from TIP range

        Returns:
            Dict with:
            - nodes: List of node info for querying
            - hash_range: (hash_min, hash_max)
            - coverage: Estimated fraction of hash space covered
        """
        nodes = self.get_nodes_for_tip_range(hash_min, hash_max)

        # Calculate coverage estimate
        range_size = hash_max - hash_min
        coverage = range_size / self.hash_space_size if self.hash_space_size > 0 else 0.0

        # Deduplicate by physical node (multiple vnodes per physical node)
        unique_physical_nodes = {}
        for node in nodes:
            key = node['physical_node']
            if key not in unique_physical_nodes:
                unique_physical_nodes[key] = {
                    'ip': node['ip'],
                    'port': node['port'],
                    'is_local': node['is_local'],
                    'vnodes': []
                }
            unique_physical_nodes[key]['vnodes'].append({
                'vid': node['vid'],
                'hash_range': node['hash_range']
            })

        return {
            'hash_range': (hash_min, hash_max),
            'coverage': coverage,
            'nodes': list(unique_physical_nodes.values()),
            'total_vnodes': len(nodes),
            'total_physical_nodes': len(unique_physical_nodes)
        }

    def update_routing_metadata(self, tip_hash: int, entity_id: str,
                                postgres_node: str, metadata: Optional[Dict] = None):
        """
        Update routing table with metadata from CDC events.

        This is called by the CDC consumer when entities are written to PostgreSQL.
        The DLHT uses this lightweight metadata to route queries efficiently.

        Args:
            tip_hash: Hash of the TIP tag
            entity_id: ID of the entity
            postgres_node: Which PostgreSQL shard/schema has this entity
            metadata: Optional additional metadata (timestamp, size, etc.)
        """
        with self.routing_table_lock:
            # Initialize list if this hash hasn't been seen before
            if tip_hash not in self.routing_table:
                self.routing_table[tip_hash] = []

            # Check if this entity_id already exists for this hash
            existing_idx = None
            for i, entry in enumerate(self.routing_table[tip_hash]):
                if entry['entity_id'] == entity_id:
                    existing_idx = i
                    break

            entry = {
                'entity_id': entity_id,
                'postgres_node': postgres_node,
                'metadata': metadata or {},
                'updated_at': time.time()
            }

            if existing_idx is not None:
                # Update existing entry
                self.routing_table[tip_hash][existing_idx] = entry
            else:
                # Add new entry
                self.routing_table[tip_hash].append(entry)

                # Track new key insertion for adaptive training
                self.training_manager.on_key_inserted(float(tip_hash))

        logger.debug(f"Updated routing: tip_hash={tip_hash} → {postgres_node} (entity: {entity_id})")

    def query_routing_table_for_range(self, hash_min: int, hash_max: int) -> Dict[str, Any]:
        """
        Query the CDC routing table to find which PostgreSQL nodes have data
        in the given hash range.

        This is the key method for intelligent query routing - it uses the routing
        table (populated by CDC) to determine which PostgreSQL nodes actually contain
        relevant data, avoiding unnecessary queries to nodes without matching data.

        Args:
            hash_min: Minimum hash value from TIP query
            hash_max: Maximum hash value from TIP query

        Returns:
            Dict with:
            - postgres_nodes: Set of postgres node names that have data in range
            - matching_hashes: List of hashes in range
            - total_matches: Number of entities found
        """
        with self.routing_table_lock:
            matching_hashes = []
            postgres_nodes = set()
            total_entities = 0

            # Scan routing table for hashes in range
            for tip_hash, entity_list in self.routing_table.items():
                if hash_min <= tip_hash <= hash_max:
                    matching_hashes.append(tip_hash)
                    # Each hash can have multiple entities
                    for entity in entity_list:
                        postgres_nodes.add(entity['postgres_node'])
                        total_entities += 1

            logger.info(f"Routing table scan: found {total_entities} entities ({len(matching_hashes)} unique hashes) across {len(postgres_nodes)} PostgreSQL nodes")

            return {
                'postgres_nodes': list(postgres_nodes),
                'matching_hashes': matching_hashes,
                'total_matches': total_entities,
                'hash_range': (hash_min, hash_max)
            }

    def get_routing_info(self, tip_hash: int) -> Optional[List[Dict[str, Any]]]:
        """
        Get routing information for a specific TIP hash.

        Returns:
            List of dicts with postgres_node and metadata for all entities with this hash,
            or None if not found
        """
        with self.routing_table_lock:
            return self.routing_table.get(tip_hash)

    def get_routing_table_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the routing table.

        Returns:
            Stats including size, distribution across postgres nodes, etc.
        """
        with self.routing_table_lock:
            total_hashes = len(self.routing_table)
            total_entities = 0

            # Count entities per postgres node
            node_distribution = defaultdict(int)
            for entity_list in self.routing_table.values():
                for entity in entity_list:
                    node_distribution[entity['postgres_node']] += 1
                    total_entities += 1

            return {
                'total_entries': total_entities,
                'total_unique_hashes': total_hashes,
                'node_distribution': dict(node_distribution),
                'memory_size_mb': total_entities * 0.001  # Rough estimate
            }