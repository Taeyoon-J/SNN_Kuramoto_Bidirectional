import pytest
from .resource_shared_v3 import required_free_mib

def test_busy_gpu_with17GiB_free_is_admitted_but2GiB_is_not():
    threshold=required_free_mib(24576)
    assert threshold==8192
    assert 24576-7304>=threshold
    assert 24576-22106<threshold
    assert threshold>3*2234

def test_observed_usage_growth_increases_reservation():
    assert required_free_mib(24576,4000)==13024
    with pytest.raises(ValueError):required_free_mib(0)
