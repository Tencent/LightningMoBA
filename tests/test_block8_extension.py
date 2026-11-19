"""Verify shared-extension symbols, CUDA streams, devices and all warp sizes."""
import os
import pytest
import torch
import lightning_moba_cuda as extension
from lightning_moba.block8.cuda import attend
from lightning_moba.block8.routing import cuda_pattern
from test_block8_migration import reference


def main():
    for name in ('moba_varlen_fwd', 'moba_dynamic_relative_threshold_compact',
                 'moba_block8_forward', 'moba_block8_layout_probe'):
        assert hasattr(extension, name), name
    print('shared extension', extension.__file__, flush=True)
    torch.manual_seed(17)
    q = torch.randn(512, 4, 128, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(512, 1, 128, device='cuda', dtype=q.dtype)
    v = torch.randn_like(k)
    expected, counts = reference(q, k, v, .9)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    for warps in (1, 2, 4, 8):
        with torch.cuda.stream(stream):
            qh, kh, vh = [x.transpose(0, 1).contiguous() for x in (q, k, v)]
            pattern = cuda_pattern(qh, kh, .9, rows=16 * warps)
            out, actual = attend(qh, kh, vh, pattern, warps, pattern_rows=16 * warps)
        stream.synchronize()
        torch.testing.assert_close(actual.T, counts)
        torch.testing.assert_close(out.transpose(0, 1).float(), expected, atol=.035, rtol=.03)

    # A forward call must work with no runtime CUDA compiler or library cache.
    previous_path = os.environ.get('PATH')
    os.environ['PATH'] = '/nonexistent'
    try:
        out, actual = attend(qh, kh, vh, pattern, 8, pattern_rows=128)
        torch.testing.assert_close(actual.T, counts)
    finally:
        if previous_path is None:
            os.environ.pop('PATH', None)
        else:
            os.environ['PATH'] = previous_path

    args = [qh, kh, vh, pattern['keep'].float(), pattern['shift'],
            pattern['offsets'], pattern['nnz'], pattern['indices'],
            pattern['work_slot'], pattern['work_tile'], 8]
    try:
        extension.moba_block8_forward(*args)
    except RuntimeError as error:
        assert 'dtype' in str(error)
    else:
        raise AssertionError('binding accepted a float selection mask')

    if torch.cuda.device_count() > 1:
        # The binding must use the input device and restore the caller device.
        tensors = [x.to('cuda:1') for x in (qh, kh, vh)]
        with torch.cuda.device(1):
            p = cuda_pattern(tensors[0], tensors[1], .9, rows=64)
        assert torch.cuda.current_device() == 0
        out, actual = attend(*tensors, p)
        assert torch.cuda.current_device() == 0
        torch.testing.assert_close(actual.cpu().T, counts.cpu())
        torch.testing.assert_close(out.cpu().transpose(0, 1).float(), expected.cpu(), atol=.035, rtol=.03)
    print('shared symbols, 4 warp sizes, non-default stream, device guard and dtype checks passed', flush=True)


@pytest.mark.cuda
@pytest.mark.sm90
def test_block8_extension():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9,0):
        pytest.skip('block8 extension requires an SM90 GPU')
    main()


if __name__ == '__main__':
    main()
