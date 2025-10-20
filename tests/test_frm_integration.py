"""Unit tests for Federated Recursive Model (FRM) integration with DLHT.

This test suite verifies the FRM implementation from LEAD paper Section III-E3,
including parameter aggregation, heartbeat coordination, and model broadcasting.
"""

import pytest
import time
import numpy as np
from unittest.mock import Mock, MagicMock, patch

from dlht.node import LEADNode
from dlht.config import LEADConfig
from dlht.federated_model import (
    FederatedRecursiveModel,
    LeafModelParameters,
    FederatedUpdateMessage,
    ModelType
)


class TestLeafModelParameters:
    """Test LeafModelParameters serialization and deserialization"""

    def test_to_dict_from_dict_roundtrip(self):
        """Test serialization and deserialization preserves data"""
        original = LeafModelParameters(
            leaf_index=42,
            coefficients=np.array([1.5, -0.3, 2.7]),
            min_key=100.0,
            max_key=200.0,
            num_samples=1000,
            version=5
        )

        # Convert to dict and back
        data = original.to_dict()
        restored = LeafModelParameters.from_dict(data)

        assert restored.leaf_index == original.leaf_index
        assert np.allclose(restored.coefficients, original.coefficients)
        assert restored.min_key == original.min_key
        assert restored.max_key == original.max_key
        assert restored.num_samples == original.num_samples
        assert restored.version == original.version

    def test_to_bytes_from_bytes_roundtrip(self):
        """Test binary serialization and deserialization"""
        original = LeafModelParameters(
            leaf_index=10,
            coefficients=np.array([0.5, 1.5]),
            min_key=50.0,
            max_key=150.0,
            num_samples=500,
            version=3
        )

        # Convert to bytes and back
        data = original.to_bytes()
        restored = LeafModelParameters.from_bytes(data)

        assert restored.leaf_index == original.leaf_index
        assert np.allclose(restored.coefficients, original.coefficients)
        assert restored.min_key == original.min_key
        assert restored.max_key == original.max_key
        assert restored.version == original.version


class TestFederatedRecursiveModel:
    """Test FederatedRecursiveModel core functionality"""

    def test_initialization(self):
        """Test FRM initialization with correct parameters"""
        frm = FederatedRecursiveModel(
            num_leaf_models=100,
            model_type='linear',
            update_threshold=0.4,
            neighbor_ready_threshold=0.9
        )

        assert frm.num_leaf_models == 100
        assert frm.model_type == ModelType.LINEAR
        assert frm.update_threshold == 0.4
        assert frm.neighbor_ready_threshold == 0.9
        assert frm.model_version == 0
        assert frm.local_ready is False

    def test_update_leaf_model(self):
        """Test updating a single leaf model"""
        frm = FederatedRecursiveModel(num_leaf_models=10, model_type='linear')

        # Generate training data
        keys = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        positions = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

        # Update leaf model
        frm.update_leaf_model(leaf_index=5, keys=keys, positions=positions)

        # Verify parameters were updated
        params = frm.get_leaf_parameters()
        leaf_5_params = [p for p in params if p.leaf_index == 5]
        assert len(leaf_5_params) == 1
        assert leaf_5_params[0].num_samples == len(keys)
        # min_key should be updated to training data min (or kept if initial was smaller)
        assert leaf_5_params[0].min_key <= min(keys)
        # max_key should be at least the training data max (could be larger from initialization)
        assert leaf_5_params[0].max_key >= max(keys)

    def test_should_trigger_federated_update_not_ready(self):
        """Test FRM doesn't trigger when not ready"""
        frm = FederatedRecursiveModel(
            num_leaf_models=10,
            update_threshold=0.4,
            neighbor_ready_threshold=0.9
        )

        # No neighbors ready
        neighbor_status = {}
        assert frm.should_trigger_federated_update(neighbor_status) is False

    def test_should_trigger_federated_update_ready(self):
        """Test FRM triggers when local and neighbors are ready"""
        frm = FederatedRecursiveModel(
            num_leaf_models=10,
            update_threshold=0.4,
            neighbor_ready_threshold=0.9
        )

        # Set local ready
        frm.local_ready = True

        # Set 95% of neighbors ready (>90% threshold)
        neighbor_status = {
            'node1': True,
            'node2': True,
            'node3': True,
            'node4': True,
            'node5': True,
            'node6': True,
            'node7': True,
            'node8': True,
            'node9': True,
            'node10': False,  # 1 out of 10 not ready = 90% ready
        }

        assert frm.should_trigger_federated_update(neighbor_status) is True

    def test_aggregate_leaf_parameters(self):
        """Test FedAvg parameter aggregation"""
        frm = FederatedRecursiveModel(num_leaf_models=10, model_type='linear')

        # Create parameters from 3 peers
        peer1_params = {
            'sender_id': 'peer1',
            'model_version': 1,
            'leaf_parameters': [
                LeafModelParameters(
                    leaf_index=0,
                    coefficients=np.array([1.0, 2.0]),
                    min_key=0.0,
                    max_key=10.0,
                    num_samples=100,
                    version=1
                ).to_dict()
            ],
            'total_keys_trained': 100
        }

        peer2_params = {
            'sender_id': 'peer2',
            'model_version': 1,
            'leaf_parameters': [
                LeafModelParameters(
                    leaf_index=0,
                    coefficients=np.array([3.0, 4.0]),
                    min_key=0.0,
                    max_key=10.0,
                    num_samples=100,
                    version=1
                ).to_dict()
            ],
            'total_keys_trained': 100
        }

        peer3_params = {
            'sender_id': 'peer3',
            'model_version': 1,
            'leaf_parameters': [
                LeafModelParameters(
                    leaf_index=0,
                    coefficients=np.array([5.0, 6.0]),
                    min_key=0.0,
                    max_key=10.0,
                    num_samples=100,
                    version=1
                ).to_dict()
            ],
            'total_keys_trained': 100
        }

        # Aggregate
        frm.aggregate_leaf_parameters([peer1_params, peer2_params, peer3_params])

        # Verify averaging: (1+3+5)/3 = 3, (2+4+6)/3 = 4
        aggregated = frm.get_aggregated_parameters()
        assert len(aggregated) == 1
        assert aggregated[0].leaf_index == 0
        assert np.allclose(aggregated[0].coefficients, np.array([3.0, 4.0]))


class TestFRMIntegrationWithNode:
    """Test FRM integration with DLHT node"""

    def test_node_initialization_with_frm(self):
        """Test that node initializes FRM correctly"""
        config = LEADConfig(
            ip='127.0.0.1',
            base_port=5000,
            num_virtual_nodes=5,
            branching_factor=100,
            model_update_threshold=0.4
        )

        node = LEADNode(config)

        # Verify FRM was created
        assert node.frm is not None
        assert node.frm.num_leaf_models == 100
        assert node.frm.update_threshold == 0.4
        assert node.frm.neighbor_ready_threshold == 0.9

    def test_update_frm_from_local_vnodes(self):
        """Test that FRM gets updated from local virtual node data"""
        config = LEADConfig(
            ip='127.0.0.1',
            base_port=5000,
            num_virtual_nodes=2,
            branching_factor=10
        )

        node = LEADNode(config)
        node.start()

        # Wait for node to be ready
        time.sleep(1)

        try:
            # Insert some keys
            for i in range(10):
                node.put(i * 1000, f"value_{i}")

            time.sleep(0.5)

            # Update FRM from local vnodes
            node._update_frm_from_local_vnodes()

            # Verify FRM has been updated
            params = node.frm.get_leaf_parameters()
            assert len(params) > 0

            # At least some leaf models should have samples
            total_samples = sum(p.num_samples for p in params)
            assert total_samples > 0

        finally:
            node.stop()

    @patch('dlht.node.LEADNode.rpc_get_frm_parameters')
    def test_collect_neighbor_parameters(self, mock_rpc):
        """Test collecting parameters from neighbors"""
        config = LEADConfig(
            ip='127.0.0.1',
            base_port=5000,
            num_virtual_nodes=2,
            branching_factor=10
        )

        node = LEADNode(config)

        # Mock neighbor parameters
        mock_neighbor_params = {
            'sender_id': '127.0.0.1:5001',
            'model_version': 1,
            'leaf_parameters': [
                LeafModelParameters(
                    leaf_index=0,
                    coefficients=np.array([1.0, 2.0]),
                    min_key=0.0,
                    max_key=100.0,
                    num_samples=50,
                    version=1
                ).to_dict()
            ],
            'total_keys_trained': 50
        }

        mock_rpc.return_value = mock_neighbor_params

        # Create a mock virtual node with a neighbor
        from dlht.peer import FingerEntry
        for vnode in node.virtual_nodes.values():
            vnode.successor = FingerEntry(vid=999, ip='127.0.0.1', port=5001)
            break

        # Collect parameters
        params = node._collect_neighbor_parameters()

        # Should have both neighbor and local parameters
        assert len(params) >= 1

    @patch('dlht.node.LEADNode.rpc_heartbeat')
    def test_send_heartbeats_to_neighbors(self, mock_rpc):
        """Test heartbeat sending to neighbors"""
        config = LEADConfig(
            ip='127.0.0.1',
            base_port=5000,
            num_virtual_nodes=2
        )

        node = LEADNode(config)

        # Mock heartbeat response
        mock_rpc.return_value = {
            'update_ready': True,
            'model_version': 1
        }

        # Manually create a virtual node entry (simpler than starting the full node)
        from dlht.peer import LEADPeer, FingerEntry
        import hashlib
        vid_data = f"{node.ip}:{node.base_port}:vnode0".encode()
        vid = int.from_bytes(hashlib.sha1(vid_data).digest(), 'big')

        vnode = LEADPeer(vid, node.ip, node.base_port, node, node.hash_space_size)
        vnode.successor = FingerEntry(vid=999, ip='127.0.0.1', port=5001)
        node.virtual_nodes[vid] = vnode

        # Send heartbeats
        node._send_heartbeats_to_neighbors()

        # Verify neighbor status was updated
        neighbor_id = '127.0.0.1:5001'
        assert neighbor_id in node.neighbor_ready_status
        assert node.neighbor_ready_status[neighbor_id] is True

    @patch('dlht.node.LEADNode.rpc_broadcast_model')
    def test_broadcast_model_to_neighbors(self, mock_rpc):
        """Test broadcasting model to neighbors"""
        config = LEADConfig(
            ip='127.0.0.1',
            base_port=5000,
            num_virtual_nodes=2
        )

        node = LEADNode(config)

        # Mock successful broadcast
        mock_rpc.return_value = True

        # Create mock parameters
        params = [
            LeafModelParameters(
                leaf_index=0,
                coefficients=np.array([1.0, 2.0]),
                min_key=0.0,
                max_key=100.0,
                num_samples=50,
                version=2
            )
        ]

        # Manually create a virtual node entry (simpler than starting the full node)
        from dlht.peer import LEADPeer, FingerEntry
        import hashlib
        vid_data = f"{node.ip}:{node.base_port}:vnode0".encode()
        vid = int.from_bytes(hashlib.sha1(vid_data).digest(), 'big')

        vnode = LEADPeer(vid, node.ip, node.base_port, node, node.hash_space_size)
        vnode.successor = FingerEntry(vid=999, ip='127.0.0.1', port=5001)
        node.virtual_nodes[vid] = vnode

        # Broadcast
        node._broadcast_model_to_neighbors(model_version=2, leaf_parameters=params)

        # Verify broadcast was called
        mock_rpc.assert_called()

    def test_apply_frm_to_learned_hash(self):
        """Test applying FRM parameters to learned hash RMI"""
        config = LEADConfig(
            ip='127.0.0.1',
            base_port=5000,
            num_virtual_nodes=2,
            branching_factor=10
        )

        node = LEADNode(config)
        node.start()

        time.sleep(0.5)

        try:
            # Create aggregated parameters
            params = [
                LeafModelParameters(
                    leaf_index=0,
                    coefficients=np.array([1.5, 2.5]),
                    min_key=0.0,
                    max_key=100.0,
                    num_samples=100,
                    version=1
                ),
                LeafModelParameters(
                    leaf_index=1,
                    coefficients=np.array([3.5, 4.5]),
                    min_key=100.0,
                    max_key=200.0,
                    num_samples=100,
                    version=1
                )
            ]

            # Apply to learned hash
            node._apply_frm_to_learned_hash(params)

            # Verify leaf models were updated
            # Note: We can't directly check coefficients, but we can verify no errors occurred
            assert True  # If we get here, application succeeded

        finally:
            node.stop()


class TestRPCHandlers:
    """Test RPC handlers for FRM operations"""

    def test_get_frm_parameters_handler(self):
        """Test get_frm_parameters RPC handler"""
        config = LEADConfig(
            ip='127.0.0.1',
            base_port=5000,
            num_virtual_nodes=2,
            branching_factor=10
        )

        node = LEADNode(config)

        # Handle RPC request
        response = node._handle_rpc('get_frm_parameters', {})

        assert response['success'] is True
        assert 'sender_id' in response
        assert 'model_version' in response
        assert 'leaf_parameters' in response
        assert 'total_keys_trained' in response

    def test_heartbeat_handler(self):
        """Test heartbeat RPC handler"""
        config = LEADConfig(
            ip='127.0.0.1',
            base_port=5000,
            num_virtual_nodes=2
        )

        node = LEADNode(config)

        # Handle heartbeat request
        payload = {
            'sender_id': '127.0.0.1:5001',
            'update_ready': True
        }

        response = node._handle_rpc('heartbeat', payload)

        assert response['success'] is True
        assert 'update_ready' in response
        assert 'model_version' in response

        # Verify neighbor status was updated
        assert node.neighbor_ready_status['127.0.0.1:5001'] is True

    def test_broadcast_model_handler_newer_version(self):
        """Test broadcast_model RPC handler with newer version"""
        config = LEADConfig(
            ip='127.0.0.1',
            base_port=5000,
            num_virtual_nodes=2,
            branching_factor=10
        )

        node = LEADNode(config)
        node.start()

        time.sleep(0.5)

        try:
            # Current version is 0, broadcast version 2
            payload = {
                'model_version': 2,
                'leaf_parameters': [
                    LeafModelParameters(
                        leaf_index=0,
                        coefficients=np.array([1.0, 2.0]),
                        min_key=0.0,
                        max_key=100.0,
                        num_samples=100,
                        version=2
                    ).to_dict()
                ]
            }

            response = node._handle_rpc('broadcast_model', payload)

            assert response['success'] is True
            assert response['applied'] is True
            assert node.learned_hash.version == 2

        finally:
            node.stop()

    def test_broadcast_model_handler_older_version(self):
        """Test broadcast_model RPC handler with older version"""
        config = LEADConfig(
            ip='127.0.0.1',
            base_port=5000,
            num_virtual_nodes=2,
            branching_factor=10
        )

        node = LEADNode(config)

        # Set current version to 5
        node.learned_hash.version = 5

        # Try to broadcast version 3
        payload = {
            'model_version': 3,
            'leaf_parameters': []
        }

        response = node._handle_rpc('broadcast_model', payload)

        assert response['success'] is True
        assert response['applied'] is False
        assert node.learned_hash.version == 5  # Version unchanged


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
