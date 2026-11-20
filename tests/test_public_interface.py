"""Regression coverage for version dispatch and the public extension contract."""
import builtins
import importlib
import inspect

import pytest
import torch

import lightning_moba
from lightning_moba._compat import supports_custom_ops
from lightning_moba import lightning_moba_interface as interface


@pytest.mark.parametrize('version,expected', [
    ('2.3.1+cu121', False), ('2.4.0', True), ('2.9.0', True),
    ('2.10.0+cu128', True), ('2.11.0a0+git123', True),
])
def test_torch_version_dispatch(version, expected):
    assert supports_custom_ops(version) is expected


def test_modern_custom_ops_are_registered():
    assert interface._TORCH_HAS_CUSTOM_OP == supports_custom_ops(torch.__version__)
    if not interface._TORCH_HAS_CUSTOM_OP:
        pytest.skip('custom-op registration requires PyTorch >= 2.4')
    from torch._subclasses.fake_tensor import FakeTensorMode
    assert isinstance(interface._wrapped_moba_fused_topk, torch._ops.OpOverloadPacket)
    with FakeTensorMode():
        q = torch.empty(256, 4, 128, dtype=torch.bfloat16)
        km = torch.empty(4, 1, 128, dtype=q.dtype)
        cu = torch.tensor([0, 256], dtype=torch.int32)
        cum = torch.tensor([0, 4], dtype=torch.int32)
        result = interface._wrapped_moba_fused_topk(q, km, cu, cu, cum, 256, 256, 4, 64, True)
        assert len(result) == 7
        assert result[0].shape == (1, 4, 4)
        assert result[4].shape == (256, 4, 16)


def test_attention_default_matches_native_lower_bound():
    defaults = inspect.signature(lightning_moba.lightning_moba_attn_varlen_func).parameters
    assert defaults['lg_block_m'].default == 128


def test_default_entry_requires_alpha_and_exports_explicit_baseline():
    parameters = inspect.signature(lightning_moba.lightning_moba_varlen_func).parameters
    assert parameters['alpha'].kind == inspect.Parameter.KEYWORD_ONLY
    assert parameters['alpha'].default is inspect.Parameter.empty
    assert 'moba_topk' not in parameters
    assert parameters['moba_chunk_size'].default == 128
    assert callable(lightning_moba.lightning_moba_topk_varlen_func)


def test_block128_routing_is_public():
    for name in (
        'flash_dynamic_mean_pool',
        'flash_cuda_dynamic_relative_threshold_compact_varlen_func',
        'flash_cuda_dynamic_relative_threshold_padded_varlen_func',
        'flash_cuda_compact_csc_qcsr_from_bitmask',
    ):
        assert callable(getattr(lightning_moba, name))
    parameters = inspect.signature(lightning_moba.lightning_moba_cuda_kv_centric_func).parameters
    assert parameters['topk'].default is None
    assert parameters['backend'].kind == inspect.Parameter.KEYWORD_ONLY
    assert 'q_offsets' in parameters and 'q_csc' in parameters


@pytest.mark.parametrize('value', [.1, -1., float('nan'), float('inf')])
def test_unsupported_hybrid_is_rejected_before_dispatch(value):
    with pytest.raises(NotImplementedError, match='hybrid_lambda'):
        interface.flash_topk_varlen_func(None, None, None, None, None, None, None, None, hybrid_lambda=value)


def test_import_errors_are_visible(monkeypatch):
    original = builtins.__import__

    def reject_backend(name, *args, **kwargs):
        if name == 'lightning_moba.triton_kv_centric':
            raise ImportError('backend import sentinel')
        return original(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, '__import__', reject_backend)
        with pytest.raises(ImportError, match='backend import sentinel'):
            importlib.reload(lightning_moba)
    importlib.reload(lightning_moba)
    assert callable(lightning_moba.lightning_moba_cuda_kv_centric_func)


@pytest.mark.cuda
def test_default_attention_forward_backward():
    if not torch.cuda.is_available():
        pytest.skip('CUDA attention regression requires a GPU')
    torch.manual_seed(29)
    q = torch.randn(256, 4, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(256, 1, 128, device='cuda', dtype=q.dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    cu = torch.tensor([0,256], device=q.device, dtype=torch.int32)
    offsets = (torch.arange(16, device=q.device, dtype=torch.int64) * 256).reshape(1,4,4)
    nnz = torch.full((1,4,4),256,device=q.device,dtype=torch.int32)
    indices = torch.arange(256,device=q.device,dtype=torch.int32).repeat(16)
    out = interface.lightning_moba_attn_varlen_func(q,k,v,cu,cu,256,256,offsets,nnz,indices,causal=True)
    reference = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0,1)[None],k.transpose(0,1)[None],v.transpose(0,1)[None],is_causal=True,enable_gqa=True)[0].transpose(0,1)
    torch.testing.assert_close(out.float(),reference.float(),atol=.035,rtol=.03)
    grad = torch.randn_like(out)
    actual = torch.autograd.grad(out,(q,k,v),grad)
    expected = torch.autograd.grad(reference,(q,k,v),grad)
    for a,e in zip(actual,expected):
        torch.testing.assert_close(a.float(),e.float(),atol=.06,rtol=.04)


@pytest.mark.cuda
def test_zero_hybrid_scores_with_current_extension():
    if not torch.cuda.is_available():
        pytest.skip('top-k regression requires a GPU')
    q = torch.randn(256,4,128,device='cuda',dtype=torch.bfloat16)
    k = torch.randn(256,1,128,device='cuda',dtype=q.dtype)
    cu = torch.tensor([0,256],device=q.device,dtype=torch.int32)
    result = interface.flash_topk_varlen_func(q,k,cu,cu,256,256,4,128,causal=True,return_scores=True)
    assert len(result)==9
    assert not bool(torch.count_nonzero(result[-1]))
    assert not bool(torch.count_nonzero(result[-2]))
