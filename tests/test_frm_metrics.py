"""Unit tests for FRM metrics collection.

This test suite verifies that FRM metrics are properly collected and reported
during federated model updates.
"""

import pytest
import time
import numpy as np

from dlht.federated_model import FRMMetrics, FederatedRecursiveModel, LeafModelParameters


class TestFRMMetrics:
    """Test FRM metrics collection"""

    def test_initial_metrics(self):
        """Test that metrics are initialized correctly"""
        metrics = FRMMetrics()

        assert metrics.total_updates == 0
        assert metrics.total_leaf_updates == 0
        assert metrics.avg_update_latency_ms == 0.0
        assert metrics.coordinator_count == 0
        assert metrics.parameters_sent_bytes == 0
        assert metrics.parameters_received_bytes == 0
        assert metrics.neighbor_participation_rate == 0.0
        assert metrics.model_convergence_score == 0.0

    def test_record_leaf_update(self):
        """Test recording leaf model updates"""
        metrics = FRMMetrics()

        # Record 5 leaf updates
        for _ in range(5):
            metrics.record_leaf_update()

        assert metrics.total_leaf_updates == 5

    def test_record_update(self):
        """Test recording federated update"""
        metrics = FRMMetrics()

        # Record a federated update
        metrics.record_update(
            latency_ms=150.0,
            coordinator=True,
            peers_participated=5,
            total_peers=10,
            bytes_sent=1024,
            bytes_received=2048
        )

        assert metrics.total_updates == 1
        assert metrics.coordinator_count == 1
        assert metrics.avg_update_latency_ms == 150.0
        assert metrics.parameters_sent_bytes == 1024
        assert metrics.parameters_received_bytes == 2048
        assert metrics.neighbor_participation_rate == 0.5  # 5/10

    def test_record_multiple_updates(self):
        """Test averaging over multiple updates"""
        metrics = FRMMetrics()

        # Record 3 updates with different latencies
        metrics.record_update(
            latency_ms=100.0,
            coordinator=True,
            peers_participated=5,
            total_peers=10,
            bytes_sent=1000,
            bytes_received=2000
        )

        metrics.record_update(
            latency_ms=200.0,
            coordinator=False,
            peers_participated=8,
            total_peers=10,
            bytes_sent=1500,
            bytes_received=3000
        )

        metrics.record_update(
            latency_ms=150.0,
            coordinator=True,
            peers_participated=7,
            total_peers=10,
            bytes_sent=1200,
            bytes_received=2500
        )

        assert metrics.total_updates == 3
        assert metrics.coordinator_count == 2  # 2 out of 3
        assert metrics.avg_update_latency_ms == 150.0  # (100+200+150)/3
        assert metrics.parameters_sent_bytes == 3700  # 1000+1500+1200
        assert metrics.parameters_received_bytes == 7500  # 2000+3000+2500

        # Participation rate: (5+8+7)/30 = 20/30 = 0.667
        assert abs(metrics.neighbor_participation_rate - 0.667) < 0.01

    def test_convergence_score(self):
        """Test convergence score tracking"""
        metrics = FRMMetrics()

        # Update convergence score
        metrics.update_convergence_score(0.85)
        assert metrics.model_convergence_score == 0.85

        # Test clamping to [0, 1]
        metrics.update_convergence_score(1.5)
        assert metrics.model_convergence_score == 1.0

        metrics.update_convergence_score(-0.5)
        assert metrics.model_convergence_score == 0.0

    def test_to_dict(self):
        """Test metrics serialization"""
        metrics = FRMMetrics()

        metrics.record_update(
            latency_ms=125.5,
            coordinator=True,
            peers_participated=3,
            total_peers=5,
            bytes_sent=512,
            bytes_received=1024
        )
        metrics.record_leaf_update()
        metrics.update_convergence_score(0.92)

        data = metrics.to_dict()

        assert data['total_updates'] == 1
        assert data['total_leaf_updates'] == 1
        assert data['avg_update_latency_ms'] == 125.5
        assert data['last_update_time'] is not None
        assert data['coordinator_count'] == 1
        assert data['parameters_sent_bytes'] == 512
        assert data['parameters_received_bytes'] == 1024
        assert data['neighbor_participation_rate'] == 0.6
        assert data['model_convergence_score'] == 0.92
        assert 'throughput_updates_per_hour' in data

    def test_metrics_in_frm(self):
        """Test that FRM integrates metrics correctly"""
        frm = FederatedRecursiveModel(num_leaf_models=10, model_type='linear')

        # Verify metrics object exists
        assert frm.metrics is not None
        assert isinstance(frm.metrics, FRMMetrics)

        # Update a leaf model
        keys = np.array([1.0, 2.0, 3.0])
        positions = np.array([0.0, 0.5, 1.0])
        frm.update_leaf_model(leaf_index=5, keys=keys, positions=positions)

        # Verify metric was recorded
        assert frm.metrics.total_leaf_updates == 1

    def test_metrics_window_size(self):
        """Test that metrics use rolling window for averages"""
        metrics = FRMMetrics()

        # Record 150 updates (more than the 100 window size)
        for i in range(150):
            metrics.record_update(
                latency_ms=float(i),
                coordinator=(i % 2 == 0),
                peers_participated=5,
                total_peers=10,
                bytes_sent=100,
                bytes_received=200
            )

        # Average should only consider last 100 values: mean(50..149) = 99.5
        assert abs(metrics.avg_update_latency_ms - 99.5) < 0.1

        # Total updates should still be 150
        assert metrics.total_updates == 150


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
