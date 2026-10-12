"""Verify the MPS compatibility bridge against native CPU pooling."""
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('hatchfinder')

from hatchfinder_mps import adaptive_pool


def test_cpu_pool_preserves_values_and_gradient():
    source = torch.arange(117, dtype=torch.float32).reshape(1,1,9,13).requires_grad_()
    expected = torch.nn.functional.adaptive_avg_pool2d(source, (6,6))
    actual = adaptive_pool(source, 6)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    assert source.grad is not None
    assert torch.isfinite(source.grad).all()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS hardware required')
def test_nondivisible_mps_pool_matches_cpu_without_resizing():
    source = torch.arange(117, dtype=torch.float32).reshape(1,1,9,13)
    expected = torch.nn.functional.adaptive_avg_pool2d(source, (6,6))
    actual = adaptive_pool(source.to('mps'), 6).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
