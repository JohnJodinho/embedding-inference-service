"""Metrics output and aggregation tests."""

import pytest

from app.metrics import Metrics


def test_metrics_export_model_latency_histograms_and_queue_depth():
    metrics = Metrics()
    metrics.set_queue_depth("bge-base-768-v1", 3)
    metrics.observe_request_duration("bge-base-768-v1", 25)
    metrics.observe_scheduler_durations("bge-base-768-v1", 100, 250)

    output = metrics.prometheus_text()

    assert 'embedding_queue_depth{model="bge-base-768-v1"} 3' in output
    assert (
        'embedding_request_duration_seconds_bucket{model="bge-base-768-v1",le="0.025"} 1'
        in output
    )
    assert (
        'embedding_queue_wait_seconds_sum{model="bge-base-768-v1"} 0.1'
        in output
    )
    assert (
        'embedding_inference_duration_seconds_count{model="bge-base-768-v1"} 1'
        in output
    )
    assert "request_id" not in output
    assert "client_label" not in output


def test_metrics_reject_invalid_queue_depth_or_model():
    metrics = Metrics()
    with pytest.raises(ValueError):
        metrics.set_queue_depth("unknown", 1)
    with pytest.raises(ValueError):
        metrics.set_queue_depth("bge-base-768-v1", -1)
