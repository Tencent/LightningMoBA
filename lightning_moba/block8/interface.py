"""Public token-major block8 API, independent of experiment directories."""
import math
import torch
from .csc import csc_threshold_attention
from .routing import validate_alpha, cuda_pattern


def _dense_suffix(q,k,v,backend):
    if backend not in ('auto','flash','sdpa'):raise ValueError('tail_backend must be auto/flash/sdpa')
    if backend!='sdpa' and torch.cuda.get_device_capability(q.device)==(9,0):
        try:
            from vllm.vllm_flash_attn import flash_attn_varlen_func
        except ImportError:
            if backend=='flash':raise RuntimeError('flash tail requires vLLM FA3')
        else:
            sq,sk=q.shape[0],k.shape[0]
            cuq=torch.tensor([0,sq],device=q.device,dtype=torch.int32)
            cuk=torch.tensor([0,sk],device=q.device,dtype=torch.int32)
            return flash_attn_varlen_func(q.contiguous(),k.contiguous(),v.contiguous(),
                max_seqlen_q=sq,cu_seqlens_q=cuq,max_seqlen_k=sk,cu_seqlens_k=cuk,causal=True,fa_version=3)
    elif backend=='flash':raise ValueError('flash tail is verified on SM90 only')
    positions=torch.arange(k.shape[0]-q.shape[0],k.shape[0],device=q.device)
    mask=torch.arange(k.shape[0],device=q.device)[None,:]<=positions[:,None]
    return torch.nn.functional.scaled_dot_product_attention(q.transpose(0,1)[None],k.transpose(0,1)[None],v.transpose(0,1)[None],attn_mask=mask,enable_gqa=True)[0].transpose(0,1)


def lightning_moba_block8_func(q,k,v,alpha,*,backend='cuda',tail_backend='auto',return_counts=False,sink=1,window=2):
    """Causal inference attention with Q/K/V [tokens, heads, dim].

    alpha: one scalar in (0,1], shared across all heads and model layers.
    sink/window count logical 8-token blocks, exactly as the original selector.
    No per-row budget cap or minimum-budget padding. Q must be
    a suffix of K. K's unaligned tail is dense and included in token counts.
    CUDA backend is BF16/D128/SM90; Triton supports FP16/BF16, D32/64/128/256.
    """
    if backend not in ('triton','cuda'):raise ValueError('backend must be triton or cuda')
    alpha=validate_alpha(alpha)
    if not isinstance(sink,int) or not isinstance(window,int) or sink<0 or window<1:raise ValueError('sink >= 0 and window >= 1 required')
    if backend=='triton' and (sink!=1 or window!=2):raise ValueError('Triton reference currently requires sink=1/window=2')
    if tail_backend not in ('auto','flash','sdpa'):raise ValueError('invalid tail backend')
    if q.ndim!=3 or k.ndim!=3 or k.shape!=v.shape:raise ValueError('expected [tokens,heads,dim] with matching K/V')
    sq,h,d=q.shape;sk,hk,dk=k.shape
    if not q.is_cuda or any(x.device!=q.device for x in (k,v)):raise ValueError('inputs must share a CUDA device')
    if any(x.requires_grad for x in (q,k,v)):raise ValueError('block8 is inference-only')
    if q.dtype not in (torch.float16,torch.bfloat16) or k.dtype!=q.dtype or v.dtype!=q.dtype:raise ValueError('matching FP16/BF16 Q/K/V required')
    if h<1 or hk<1 or h%hk or d!=dk or d not in (32,64,128,256) or sq>sk:raise ValueError('invalid GQA dimensions or suffix lengths')
    if backend=='cuda' and (q.dtype!=torch.bfloat16 or d!=128 or torch.cuda.get_device_capability(q.device)!=(9,0)):raise ValueError('CUDA block8 requires BF16/D128/SM90')
    output=torch.empty_like(q);counts=torch.empty((sq,h),device=q.device,dtype=torch.int32)
    if sq==0:return (output,counts) if return_counts else output
    aligned=sk//256*256
    prefix=max(0,min(sq,aligned-(sk-sq)))
    if prefix:
        qh=q[:prefix].transpose(0,1).contiguous()
        kh=k[:aligned].transpose(0,1).contiguous();vh=v[:aligned].transpose(0,1).contiguous()
        kwargs=dict(block=8,width=128,group=2,rows=64,select_block_m=128)
        if backend=='triton':
            logalpha=torch.full((h,),math.log(alpha),device=q.device,dtype=torch.float32)
            out,cc=csc_threshold_attention(qh,kh,vh,logalpha,**kwargs)
        else:
            from .cuda import attend
            pattern=cuda_pattern(qh,kh,alpha,sink=sink,window=window,**kwargs)
            out,cc=attend(qh,kh,vh,pattern,4,pattern_rows=64)
        output[:prefix].copy_(out.transpose(0,1));counts[:prefix].copy_(cc.transpose(0,1))
    if prefix<sq:
        output[prefix:].copy_(_dense_suffix(q[prefix:],k,v,tail_backend))
        visible=torch.arange(sk-sq+prefix+1,sk+1,device=q.device,dtype=torch.int32)
        counts[prefix:]=visible[:,None]
    return (output,counts) if return_counts else output


def lightning_moba_block8_varlen_func(q,k,v,cu_seqlens_q,cu_seqlens_k,max_seqlen_q,max_seqlen_k,alpha,**kwargs):
    """Packed [total_tokens,heads,dim] API. Sequences dispatched independently.

    cu_seqlens metadata is synchronized to CPU. This first adapter is not
    CUDA-graph safe or a fused multi-sequence kernel. One alpha is shared by all heads/sequences.
    """
    for cu,total in ((cu_seqlens_q,q.shape[0]),(cu_seqlens_k,k.shape[0])):
        if cu.ndim!=1 or cu.dtype!=torch.int32 or cu.device!=q.device or not cu.is_contiguous():raise ValueError('cu_seqlens must be contiguous CUDA int32')
    qs=cu_seqlens_q.cpu().tolist();ks=cu_seqlens_k.cpu().tolist()
    if len(qs)!=len(ks) or len(qs)<2 or qs[0]!=0 or ks[0]!=0 or qs[-1]!=q.shape[0] or ks[-1]!=k.shape[0]:raise ValueError('invalid cumulative lengths')
    output=torch.empty_like(q);counts=torch.empty(q.shape[:2],device=q.device,dtype=torch.int32)
    options=dict(kwargs);want_counts=options.pop('return_counts',False)
    for i in range(len(qs)-1):
        nq,nk=qs[i+1]-qs[i],ks[i+1]-ks[i]
        if nq<0 or nk<0 or nq>nk or nq>max_seqlen_q or nk>max_seqlen_k:raise ValueError('invalid per-sequence/max lengths')
        out,cc=lightning_moba_block8_func(q[qs[i]:qs[i+1]],k[ks[i]:ks[i+1]],v[ks[i]:ks[i+1]],alpha,return_counts=True,**options)
        output[qs[i]:qs[i+1]]=out;counts[qs[i]:qs[i+1]]=cc
    return (output,counts) if want_counts else output
