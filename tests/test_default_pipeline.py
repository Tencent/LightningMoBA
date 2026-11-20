"""Default dispatch contracts without CUDA, plus real SM90 numerical regressions."""
import ast
from contextlib import nullcontext
import math
from numbers import Real
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def dispatch(monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'lightning_moba/lightning_moba_interface.py'
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {'lightning_moba_varlen_func', 'lightning_moba_topk_varlen_func'}]
    assert len(functions) == 2
    calls = []
    pattern = SimpleNamespace(col_offsets=object(), col_nnz=object(), csc_indices=object())
    km, cukm, q_offsets, q_csc = (object() for _ in range(4))
    partial, lse, output = (object() for _ in range(3))

    def stage(name, result):
        def run(*args, **kwargs):
            calls.append(name)
            return result
        return Mock(side_effect=run)

    pool = stage('pool', (km, cukm, 4))
    select = stage('relative_threshold', pattern)
    scatter = stage('csc_qcsr', (pattern, q_offsets, q_csc))
    pipeline = stage('sm90_pipeline', (partial, lse))
    combine = stage('combine', output)
    native = SimpleNamespace(
        moba_dynamic_relative_threshold_compact=object(),
        moba_compact_csc_qcsr_from_bitmask=object(),
        moba_sm90_phase4b_csc_pipe=pipeline,
        moba_combine_q_csr=combine,
    )
    pool_module = ModuleType('lightning_moba.triton_mean_pool')
    pool_module.flash_dynamic_mean_pool = pool
    route_module = ModuleType('lightning_moba.cuda_dynamic_threshold')
    route_module.flash_cuda_dynamic_relative_threshold_compact_varlen_func = select
    route_module.flash_cuda_compact_csc_qcsr_from_bitmask = scatter
    monkeypatch.setitem(sys.modules, pool_module.__name__, pool_module)
    monkeypatch.setitem(sys.modules, route_module.__name__, route_module)
    monkeypatch.setattr(torch.cuda, 'device', lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: False)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: (9, 0))
    monkeypatch.setattr(torch, 'empty_like', lambda tensor: output)

    def tensor(shape, dtype=torch.bfloat16, boundaries=None):
        value = Mock(spec=torch.Tensor)
        value.shape, value.ndim = shape, len(shape)
        value.dtype, value.device = dtype, torch.device('cuda:0')
        value.is_cuda, value.requires_grad = True, False
        value.is_contiguous.return_value = True
        value.contiguous.return_value = value
        value.numel.return_value = math.prod(shape)
        if boundaries is not None:
            value.cpu.return_value.tolist.return_value = boundaries
        return value

    q, k, v = [tensor(shape) for shape in [(256, 4, 128), (512, 1, 128), (512, 1, 128)]]
    cuq = tensor((2,), torch.int32, [0, 256])
    cuk = tensor((2,), torch.int32, [0, 512])
    forbidden = Mock(side_effect=AssertionError('default must not dispatch to Top-K/query-centric'))
    namespace = dict(torch=torch, math=math, Real=Real, lightning_moba_gpu=native,
                     flash_topk_varlen_func=forbidden, decide_lg_block_m=forbidden,
                     lightning_moba_attn_varlen_func=forbidden)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    return SimpleNamespace(
        run=namespace['lightning_moba_varlen_func'], args=(q, k, v, cuq, cuk, 256, 512),
        pool=pool, select=select, scatter=scatter, pipeline=pipeline, combine=combine,
        native=native, pattern=pattern, km=km, cukm=cukm, q_offsets=q_offsets,
        q_csc=q_csc, partial=partial, lse=lse, output=output, calls=calls,
        forbidden=forbidden,
    )


def test_default_calls_pipeline_without_environment_switches(dispatch, monkeypatch):
    monkeypatch.setenv('LIGHTNING_MOBA_STREAM_CHUNK_EDGES', '2000000')
    monkeypatch.setenv('LIGHTNING_MOBA_KVCENTRIC_SM90_K1', 'disabled')
    result = dispatch.run(*dispatch.args, alpha=0.75, softmax_scale=0.125)
    q, k, v, cuq, cuk, maxq, maxk = dispatch.args
    assert result is dispatch.output
    assert dispatch.calls == ['pool', 'relative_threshold', 'csc_qcsr', 'sm90_pipeline', 'combine']
    dispatch.pool.assert_called_once_with(k, cuk, maxk, 128)
    dispatch.select.assert_called_once_with(
        q, dispatch.km, cuq, cuk, dispatch.cukm, maxq, maxk, 0.75, 128,
        softmax_scale=0.125, sink=1, window=2, causal=True,
        min_blocks_per_row=0, build_csc=False,
    )
    dispatch.scatter.assert_called_once_with(dispatch.pattern, cuq, dispatch.cukm, maxq, maxk, 128)
    dispatch.pipeline.assert_called_once_with(
        q, k, v, dispatch.pattern.col_offsets, dispatch.pattern.col_nnz,
        dispatch.pattern.csc_indices, cuq, cuk, True, 0.125,
    )
    dispatch.combine.assert_called_once_with(
        dispatch.partial, dispatch.lse, dispatch.q_offsets, dispatch.q_csc, dispatch.output,
    )
    dispatch.forbidden.assert_not_called()


@pytest.mark.parametrize('alpha', [0, -1, 1.01, float('nan'), float('inf'), True, [0.5]])
def test_invalid_alpha_rejected_before_dispatch(dispatch, alpha):
    with pytest.raises(ValueError, match='alpha'):
        dispatch.run(*dispatch.args, alpha=alpha)
    assert not dispatch.calls


def test_old_topk_arguments_are_not_reinterpreted(dispatch):
    with pytest.raises(TypeError):
        dispatch.run(*dispatch.args, 128, 16)
    with pytest.raises(TypeError):
        dispatch.run(*dispatch.args, moba_topk=16, alpha=0.5)
    assert not dispatch.calls


@pytest.mark.parametrize('index', [0, 1, 2])
def test_gradient_inputs_rejected(dispatch, index):
    dispatch.args[index].requires_grad = True
    with pytest.raises(ValueError, match='inference-only'):
        dispatch.run(*dispatch.args, alpha=0.5)
    assert not dispatch.calls


@pytest.mark.parametrize('options', [
    {'moba_chunk_size': 64}, {'causal': False}, {'sink': -1}, {'window': 0},
    {'softmax_scale': 0}, {'softmax_scale': float('nan')},
])
def test_unsupported_parameters_rejected(dispatch, options):
    with pytest.raises(ValueError):
        dispatch.run(*dispatch.args, alpha=0.5, **options)
    assert not dispatch.calls


def test_unsupported_gpu_does_not_fallback(dispatch, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: (8, 0))
    with pytest.raises(ValueError, match='SM90'):
        dispatch.run(*dispatch.args, alpha=0.5)
    assert not dispatch.calls


def test_capture_rejected_before_metadata_read(dispatch, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: True)
    with pytest.raises(RuntimeError, match='eager-only'):
        dispatch.run(*dispatch.args, alpha=0.5)
    dispatch.args[3].cpu.assert_not_called()
    assert not dispatch.calls


@pytest.mark.parametrize('symbol', [
    'moba_sm90_phase4b_csc_pipe', 'moba_combine_q_csr',
    'moba_dynamic_relative_threshold_compact', 'moba_compact_csc_qcsr_from_bitmask',
])
def test_missing_extension_symbol_does_not_fallback(dispatch, symbol):
    delattr(dispatch.native, symbol)
    with pytest.raises(RuntimeError, match='rebuild'):
        dispatch.run(*dispatch.args, alpha=0.5)
    assert not dispatch.calls


@pytest.mark.parametrize('boundaries', [[1, 256], [0, 255], [0, -1, 256]])
def test_invalid_cumulative_lengths_rejected(dispatch, boundaries):
    dispatch.args[3].cpu.return_value.tolist.return_value = boundaries
    with pytest.raises(ValueError):
        dispatch.run(*dispatch.args, alpha=0.5)
    assert not dispatch.calls


def _reference(q, k, v, q_lengths, k_lengths, alpha):
    results = []
    qb = kb = 0
    for qlen, klen in zip(q_lengths, k_lengths):
        keys = k[kb:kb + klen].transpose(0, 1)
        means = torch.stack([keys[:, start:start + 128].float().mean(1)
                             for start in range(0, klen, 128)], dim=1).to(k.dtype).float()
        queries = q[qb:qb + qlen].transpose(0, 1).float()
        ratio = q.shape[1] // k.shape[1]
        scores = queries @ means.repeat_interleave(ratio, 0).transpose(-1, -2) * 128 ** -0.5
        positions = torch.arange(klen - qlen, klen, device=q.device)[:, None]
        columns = torch.arange(means.shape[1], device=q.device)[None, :]
        local = positions // 128
        forced = (columns < 1) | ((columns <= local) & (local - columns < 2))
        completed = (columns + 1) * 128 <= positions
        maximum = scores.masked_fill(~(forced | completed), -float('inf')).amax(-1, keepdim=True)
        selected = forced | (completed & (scores >= maximum + math.log(alpha)))
        tokens = torch.arange(klen, device=q.device)
        allowed = selected[:, :, tokens // 128] & (tokens[None, :] <= positions)
        full_keys = keys.repeat_interleave(ratio, 0).float()
        values = v[kb:kb + klen].transpose(0, 1).repeat_interleave(ratio, 0).float()
        logits = queries @ full_keys.transpose(-1, -2) * 128 ** -0.5
        results.append((logits.masked_fill(~allowed, -float('inf')).softmax(-1) @ values).transpose(0, 1))
        qb += qlen
        kb += klen
    return torch.cat(results)


@pytest.mark.cuda
@pytest.mark.sm90
@pytest.mark.parametrize('q_lengths,k_lengths,tied,kv_heads', [
    ([512], [512], False, 1),
    ([17, 256], [512, 1024], False, 2),
    ([17, 257], [129, 513], False, 2),
    ([1], [63], False, 1),
    ([1], [8448], True, 1),
    ([4352], [4352], True, 1),
])
def test_default_pipeline_numerics(q_lengths, k_lengths, tied, kv_heads, monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('default pipeline numerical regression requires SM90')
    import lightning_moba
    import lightning_moba_cuda

    torch.manual_seed(42)
    heads = 1 if tied else 4
    q = torch.randn(sum(q_lengths), heads, 128, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(sum(k_lengths), kv_heads, 128, device='cuda', dtype=q.dtype)
    v = torch.randn_like(k)
    if tied:
        q.zero_()
    cuq = torch.tensor([0] + list(torch.tensor(q_lengths).cumsum(0).tolist()), device='cuda', dtype=torch.int32)
    cuk = torch.tensor([0] + list(torch.tensor(k_lengths).cumsum(0).tolist()), device='cuda', dtype=torch.int32)
    pipeline = Mock(wraps=lightning_moba_cuda.moba_sm90_phase4b_csc_pipe)
    monkeypatch.setattr(lightning_moba_cuda, 'moba_sm90_phase4b_csc_pipe', pipeline)
    monkeypatch.setenv('LIGHTNING_MOBA_STREAM_CHUNK_EDGES', '2000000')
    monkeypatch.setenv('LIGHTNING_MOBA_KVCENTRIC_SM90_K1', 'disabled')
    out = lightning_moba.lightning_moba_varlen_func(
        q, k, v, cuq, cuk, max(q_lengths), max(k_lengths), alpha=0.8,
    )
    pipeline.assert_called_once()
    expected = _reference(q, k, v, q_lengths, k_lengths, 0.8)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), expected, atol=0.04, rtol=0.04)

    km, cukm, _ = lightning_moba.flash_dynamic_mean_pool(k, cuk, max(k_lengths), 128)
    pattern = lightning_moba.flash_cuda_dynamic_relative_threshold_compact_varlen_func(
        q, km, cuq, cuk, cukm, max(q_lengths), max(k_lengths), 0.8, 128, build_csc=False,
    )
    pattern, offsets, edges = lightning_moba.flash_cuda_compact_csc_qcsr_from_bitmask(
        pattern, cuq, cukm, max(q_lengths), max(k_lengths), 128,
    )
    pipeline.reset_mock()
    staged = lightning_moba.lightning_moba_cuda_kv_centric_func(
        q, k, v, cuq, cuk, max(q_lengths), max(k_lengths),
        pattern.col_offsets, pattern.col_nnz, pattern.csc_indices,
        chunk_size=128, backend='sm90', q_offsets=offsets, q_csc=edges,
    )
    pipeline.assert_called_once()
    torch.testing.assert_close(staged.float(), expected, atol=0.04, rtol=0.04)
