"""CPU checks for benchmark calibration and staged SM90 dispatch contracts."""
import ast
from contextlib import nullcontext
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('routing_sweep', ROOT / 'benchmarks/bench_routing_sweep.py')
sweep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sweep)


def test_calibration_matches_quantized_monotonic_counts():
    count = lambda alpha: round(100 - 50 * math.log(alpha))
    result = sweep.calibrate(count, 180)
    assert result['matched']
    assert result['selected_pairs'] == count(result['alpha']) == 180
    assert result['relative_pair_error'] <= 0.0005


def test_calibration_does_not_split_ties():
    result = sweep.calibrate(lambda alpha: 100 if alpha > 0.5 else 300, 200)
    assert not result['matched']
    assert result['selected_pairs'] in (100, 300)
    assert result['relative_pair_error'] == 0.5
    assert {point['pairs'] for point in result['bracket']} == {100, 300}


def test_calibration_reports_anchor_floor():
    result = sweep.calibrate(lambda alpha: 250, 100)
    assert not result['matched']
    assert result['alpha'] == 1
    assert result['evaluations'] == 1


def test_calibration_handles_dense_target():
    result = sweep.calibrate(lambda alpha: min(500, round(100 - 50 * math.log(alpha))), 500)
    assert result['matched']
    assert result['selected_pairs'] == 500


@pytest.mark.parametrize('target,tolerance,iterations', [(0, 0.1, 5), (1, 0, 5), (1, 1, 5), (1, 0.1, 0)])
def test_invalid_calibration_settings(target, tolerance, iterations):
    with pytest.raises(ValueError):
        sweep.calibrate(lambda alpha: 1, target, tolerance, iterations)


def test_csc_validation_counts_only_valid_prefixes():
    offsets = torch.tensor([[[0, 5, 5]]], dtype=torch.int64)
    counts = torch.tensor([[[2, 0, 1]]], dtype=torch.int32)
    indices = torch.tensor([0, 1, -999, -999, -999, 2, -999], dtype=torch.int32)
    assert sweep.validate_csc(offsets, counts, indices, 3) == 3
    indices[5] = 3
    with pytest.raises(RuntimeError, match='out-of-range'):
        sweep.validate_csc(offsets, counts, indices, 3)


def test_safe_capture_layout_validation(tmp_path):
    path = tmp_path / 'capture.pt'
    q = torch.zeros(4, 256, 128, dtype=torch.bfloat16)
    k = torch.zeros(2, 256, 128, dtype=torch.bfloat16)
    torch.save({'q': q, 'k': k}, path)
    loaded_q, loaded_k, digest = sweep.capture_input(path, 'head-major')
    assert loaded_q.shape == (256, 4, 128)
    assert loaded_k.shape == (256, 2, 128)
    assert len(digest) == 64
    torch.save({'q': q.float(), 'k': k.float()}, path)
    with pytest.raises(ValueError, match='BF16'):
        sweep.capture_input(path, 'head-major')


@pytest.fixture
def staged(monkeypatch):
    path = ROOT / 'lightning_moba/triton_kv_centric.py'
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == 'lightning_moba_cuda_kv_centric_func')
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), fn],
                        type_ignores=[])
    ast.fix_missing_locations(module)
    native = SimpleNamespace(moba_sm90_phase4b_csc_pipe=Mock(return_value=('partial', 'lse')),
                             moba_combine_q_csr=Mock(return_value='result'))
    monkeypatch.setitem(sys.modules, 'lightning_moba_cuda', native)
    monkeypatch.setattr(torch.cuda, 'device', lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: (9, 0))
    monkeypatch.setattr(torch, 'empty_like', lambda tensor: 'output')
    namespace = {'torch': torch}
    exec(compile(module, str(path), 'exec'), namespace)

    def tensor(shape, dtype):
        value = Mock(spec=torch.Tensor)
        value.ndim, value.shape = len(shape), shape
        value.dtype, value.device = dtype, torch.device('cuda:0')
        value.is_cuda, value.requires_grad = True, False
        value.is_contiguous.return_value = True
        value.contiguous.return_value = value
        value.numel.return_value = math.prod(shape)
        return value

    q = tensor((256, 4, 128), torch.bfloat16)
    k, v = [tensor((512, 2, 128), torch.bfloat16) for _ in range(2)]
    cuq, cuk = [tensor((2,), torch.int32) for _ in range(2)]
    offsets = tensor((1, 4, 4), torch.int64)
    counts = tensor((1, 4, 4), torch.int32)
    indices = tensor((100,), torch.int32)
    q_offsets = tensor((1025,), torch.int64)
    q_csc = tensor((100,), torch.int32)
    return SimpleNamespace(run=namespace[fn.name], native=native,
                           args=(q, k, v, cuq, cuk, 256, 512, offsets, counts, indices, 128),
                           kwargs=dict(backend='sm90', q_offsets=q_offsets, q_csc=q_csc))


def test_staged_dynamic_dispatch_has_no_topk_capacity(staged, monkeypatch):
    monkeypatch.setenv('LIGHTNING_MOBA_STREAM_CHUNK_EDGES', '2000000')
    monkeypatch.setenv('LIGHTNING_MOBA_KVCENTRIC_SM90_K1', 'disabled')
    assert staged.run(*staged.args, **staged.kwargs) == 'result'
    staged.native.moba_sm90_phase4b_csc_pipe.assert_called_once()
    staged.native.moba_combine_q_csr.assert_called_once_with(
        'partial', 'lse', staged.kwargs['q_offsets'], staged.kwargs['q_csc'], 'output')


def test_dynamic_dispatch_rejects_topk_capacity(staged):
    with pytest.raises(ValueError, match='Top-K capacity'):
        staged.run(*staged.args, topk=16, **staged.kwargs)
    staged.native.moba_sm90_phase4b_csc_pipe.assert_not_called()


def test_dynamic_dispatch_requires_matching_qcsr(staged):
    staged.kwargs['q_csc'].numel.return_value = 99
    with pytest.raises(ValueError, match='dimensions'):
        staged.run(*staged.args, **staged.kwargs)
    staged.native.moba_sm90_phase4b_csc_pipe.assert_not_called()


def test_legacy_dispatch_requires_explicit_budget(staged):
    with pytest.raises(ValueError, match='explicit positive'):
        staged.run(*staged.args)


def test_dynamic_dispatch_has_no_hardware_fallback(staged, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: (8, 0))
    with pytest.raises(ValueError, match='SM90'):
        staged.run(*staged.args, **staged.kwargs)
    staged.native.moba_sm90_phase4b_csc_pipe.assert_not_called()
