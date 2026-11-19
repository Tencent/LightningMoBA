"""Stable softmax regressions for finite inputs with large token-score spread."""
import pytest
import torch

from lightning_moba import lightning_moba_block8_func
from lightning_moba.block8.cuda import attend
from lightning_moba.block8.routing import cuda_pattern
from test_block8_migration import reference

pytestmark = [pytest.mark.cuda, pytest.mark.sm90]


def require_sm90():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('numerical regression requires SM90')


@pytest.mark.parametrize('backend', ['cuda', 'triton'])
@pytest.mark.parametrize('case', ['overflow', 'underflow', 'columns', 'future', 'common_offset'])
def test_stable_large_logits(backend, case):
    require_sm90()
    torch.manual_seed(83)
    n = 256 if case in ('overflow', 'underflow') else 512
    q = torch.full((n, 4, 128), 4., device='cuda', dtype=torch.bfloat16)
    k = torch.full((n, 1, 128), 4., device=q.device, dtype=q.dtype)
    k[1::2] = -4
    v = torch.ones_like(k)
    if case == 'underflow':
        k.neg_()  # The first visible score is -181; a pooled shift loses it.
    elif case == 'columns':
        k[256:] *= 2
        v[:256] = -2
        v[256:] = 3
    elif case == 'future':
        k.fill_(-4)
        k[-1] = 64  # Masked future scores must not enter the true maximum.
    elif case == 'common_offset':
        q.fill_(1e5)
        k.fill_(1e5)
        v.normal_()
    expected, expected_counts = reference(q, k, v, 1.)
    out, counts = lightning_moba_block8_func(q, k, v, 1., backend=backend, return_counts=True)
    assert bool(out.isfinite().all())
    torch.testing.assert_close(counts, expected_counts)
    torch.testing.assert_close(out.float(), expected, atol=.035, rtol=.03)
    if case == 'columns':
        torch.testing.assert_close(out[256:].float(), torch.full_like(out[256:].float(), 3.))


@pytest.mark.parametrize('warps', [1, 2, 4, 8])
def test_true_max_ignores_pooled_shift_and_handles_partial_tiles(warps):
    require_sm90()
    torch.manual_seed(91)
    q = torch.full((17, 4, 128), 4., device='cuda', dtype=torch.bfloat16)
    k = torch.full((512, 1, 128), 4., device=q.device, dtype=q.dtype)
    k[1::2] = -4
    k[256:] *= 2
    v = torch.randn_like(k)
    expected, expected_counts = reference(q, k, v, 1.)
    qh, kh, vh = [x.transpose(0, 1).contiguous() for x in (q, k, v)]
    p = cuda_pattern(qh, kh, 1., rows=16 * warps)
    keep = p['keep'].clone()
    # Retained for ABI compatibility; this tensor must not normalize attention.
    for shift in (-1e6, 1e6):
        p['shift'].fill_(shift)
        out, counts = attend(qh, kh, vh, p, warps, pattern_rows=16 * warps)
        assert bool(out.isfinite().all())
        torch.testing.assert_close(counts.T, expected_counts)
        torch.testing.assert_close(out.transpose(0, 1).float(), expected, atol=.035, rtol=.03)
    torch.testing.assert_close(p['keep'], keep)


@pytest.mark.parametrize('backend', ['cuda', 'triton'])
def test_large_logits_with_unaligned_suffix(backend):
    require_sm90()
    torch.manual_seed(97)
    q = torch.full((513, 4, 128), 4., device='cuda', dtype=torch.bfloat16)
    k = torch.full((777, 1, 128), 4., device=q.device, dtype=q.dtype)
    k[1::2] = -4
    v = torch.randn_like(k)
    expected, expected_counts = reference(q, k, v, 1.)
    out, counts = lightning_moba_block8_func(q, k, v, 1., backend=backend,
                                           tail_backend='sdpa', return_counts=True)
    assert bool(out.isfinite().all())
    torch.testing.assert_close(counts, expected_counts)
    torch.testing.assert_close(out.float(), expected, atol=.035, rtol=.03)


@pytest.mark.parametrize('dim', [32, 64, 128, 256])
def test_fp16_reference_with_large_scores(dim):
    require_sm90()
    torch.manual_seed(101)
    q = torch.full((17, 4, dim), 4., device='cuda', dtype=torch.float16)
    k = torch.full((512, 1, dim), 4., device=q.device, dtype=q.dtype)
    k[::2] = -4
    k[256:] *= 2
    v = torch.randn_like(k)
    expected, expected_counts = reference(q, k, v, 1.)
    out, counts = lightning_moba_block8_func(q, k, v, 1., backend='triton', return_counts=True)
    assert bool(out.isfinite().all())
    torch.testing.assert_close(counts, expected_counts)
    torch.testing.assert_close(out.float(), expected, atol=.006, rtol=.03)
