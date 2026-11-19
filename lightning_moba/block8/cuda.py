"""Block8 CUDA attention from the shared LightningMoBA extension."""
import torch

_layout_checked = set()


def extension():
    import lightning_moba_cuda
    if not all(hasattr(lightning_moba_cuda, name) for name in
               ('moba_block8_forward', 'moba_block8_layout_probe')):
        raise RuntimeError('Rebuild LightningMoBA: its CUDA extension must include block8')
    return lightning_moba_cuda


def validate_fragment_layout(reference, module):
    device = reference.device.index
    if device in _layout_checked:
        return
    if torch.cuda.get_device_capability(reference.device) != (9, 0):
        raise RuntimeError('register variant requires verified SM90 fragment layout')
    out = module.moba_block8_layout_probe(reference)
    expected = torch.empty((16, 16), dtype=torch.float32)
    for lane in range(32):
        for i in range(8):
            row = lane // 4 + (8 if i & 2 else 0)
            col = (lane % 4) * 2 + (8 if i & 4 else 0) + (i & 1)
            expected[row, col] = lane * 100 + i
    if not torch.equal(out.cpu(), expected):
        raise RuntimeError('WMMA fragment mapping changed; register variant disabled')
    _layout_checked.add(device)


def attend(q, k, v, p, warps=4, pattern_rows=64):
    if pattern_rows != 16 * warps:
        raise ValueError('CSC rows must equal 16*warps')
    if p['n_exec'] * 256 != k.shape[1]:
        raise ValueError('CSC execution columns must cover 256 tokens')
    module = extension()
    validate_fragment_layout(q, module)
    acc, den, count = module.moba_block8_forward(
        q, k, v, p['keep'], p['shift'], p['offsets'], p['nnz'],
        p['indices'], p['work_slot'], p['work_tile'], warps)
    return (acc / den.clamp_min(1e-30)[..., None]).to(q.dtype), count
