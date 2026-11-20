"""GPU integration checks, collected by pytest and runnable directly."""
import math
import pytest
import torch
import lightning_moba
from lightning_moba import lightning_moba_block8_func as block8,lightning_moba_block8_varlen_func as varlen

def reference(q,k,v,alpha,sink=1,window=2):
    sq,h,d=q.shape;sk=k.shape[0];aligned=sk//256*256;prefix=max(0,min(sq,aligned-(sk-sq)))
    tokens=torch.arange(sk,device=q.device)
    allowed=(tokens[None,:]<=torch.arange(sk-sq,sk,device=q.device)[:,None])[None].expand(h,-1,-1).clone()
    if prefix:
        # Independent pooled-score reference; do not derive expected mask from the implementation.
        kh=k[:aligned].transpose(0,1).float()
        km=kh.reshape(kh.shape[0],aligned//8,8,d).mean(2).to(k.dtype).float()
        scores=q[:prefix].transpose(0,1).float()@km.repeat_interleave(h//k.shape[1],0).transpose(-1,-2)*d**-.5
        pos=torch.arange(sk-sq,sk-sq+prefix,device=q.device)[:,None]
        cols=torch.arange(aligned//8,device=q.device)[None,:]
        local=pos//8
        forced=(cols<sink)|((cols<=local)&(local-cols<window))
        eligible=forced|((cols+1)*8<=pos)
        maximum=scores.masked_fill(~eligible,float('-inf')).amax(-1,keepdim=True)
        keep=forced|(eligible&(scores>=maximum+math.log(alpha)))
        allowed[:,:prefix,:aligned]&=keep[:,:,tokens[:aligned]//8]
    qh=q.transpose(0,1).float();kh=k.transpose(0,1).repeat_interleave(h//k.shape[1],0).float();vh=v.transpose(0,1).repeat_interleave(h//k.shape[1],0).float()
    score=qh@kh.transpose(-1,-2)*d**-.5
    return (score.masked_fill(~allowed,float('-inf')).softmax(-1)@vh).transpose(0,1),allowed.sum(-1).T.int()

def main():
    print('package',lightning_moba.__file__,flush=True)
    torch.manual_seed(42);cases=0
    for dtype,d in [(torch.float16,64),(torch.bfloat16,128)]:
        for sq,sk in [(1,17),(17,513),(777,777),(512,512)]:
            q=torch.randn(sq,4,d,device='cuda',dtype=dtype);k=torch.randn(sk,1,d,device='cuda',dtype=dtype);v=torch.randn_like(k)
            a=.9
            expected,count=reference(q,k,v,a)
            for backend in (['triton','cuda'] if d==128 else ['triton']):
                out,cc=block8(q,k,v,a,backend=backend,tail_backend='sdpa',return_counts=True)
                torch.testing.assert_close(cc,count)
                torch.testing.assert_close(out.float(),expected,atol=.035 if dtype==torch.bfloat16 else .006,rtol=.03)
                cases+=1
    # Packed sequences have independent causal positions, including a short suffix.
    q=torch.randn(530,4,128,device='cuda',dtype=torch.bfloat16);k=torch.randn(1290,1,128,device='cuda',dtype=q.dtype);v=torch.randn_like(k);a=.9
    cuq=torch.tensor([0,513,530],device='cuda',dtype=torch.int32);cuk=torch.tensor([0,777,1290],device='cuda',dtype=torch.int32)
    for backend in ['triton','cuda']:
        out,cc=varlen(q,k,v,cuq,cuk,513,777,a,backend=backend,tail_backend='sdpa',return_counts=True)
        for qb,qe,kb,ke in [(0,513,0,777),(513,530,777,1290)]:
            e,c=block8(q[qb:qe],k[kb:ke],v[kb:ke],a,backend=backend,tail_backend='sdpa',return_counts=True)
            torch.testing.assert_close(out[qb:qe].float(),e.float(),atol=.035,rtol=.03);torch.testing.assert_close(cc[qb:qe],c)
        cases+=1
    invalid=[
        lambda:block8(q,k,v,a,backend='unknown'),
        lambda:block8(q,k,v,torch.full((4,),.9,device=q.device)),
        lambda:block8(q.detach().requires_grad_(True),k,v,a),
        lambda:varlen(q,k,v,cuq.long(),cuk,513,777,a),
    ]
    invalid += [lambda value=value:block8(q,k,v,value) for value in (0.,-1.,1.1,float('nan'),float('inf'))]
    for operation in invalid:
        try:operation()
        except ValueError:pass
        else:raise AssertionError('invalid input was accepted')
    # CUDA must preserve the original configurable sink/window semantics.
    q=torch.randn(512,4,128,device='cuda',dtype=torch.bfloat16);k=torch.randn(512,1,128,device='cuda',dtype=q.dtype);v=torch.randn_like(k)
    for sink,window in [(0,1),(2,3)]:
        out,cc=block8(q,k,v,.9,sink=sink,window=window,return_counts=True)
        expected,count=reference(q,k,v,.9,sink=sink,window=window)
        torch.testing.assert_close(cc,count)
        torch.testing.assert_close(out.float(),expected,atol=.035,rtol=.03)
        cases+=1
    # One shared alpha still permits very different budgets across heads.
    # Head0 has tied scores; head1 has one dominant sink block. No Top-K/cap.
    q=torch.zeros(512,2,128,device='cuda',dtype=torch.bfloat16)
    q[:,1,0]=1
    k=torch.zeros(512,1,128,device='cuda',dtype=q.dtype);k[:8,0,0]=100
    v=torch.randn_like(k)
    out,cc=block8(q,k,v,.5,return_counts=True)
    expected,count=reference(q,k,v,.5)
    torch.testing.assert_close(cc,count)
    torch.testing.assert_close(cc[:,0],torch.arange(1,513,device=q.device,dtype=torch.int32))
    assert int(cc[-1,1])==24
    torch.testing.assert_close(out.float(),expected,atol=.035,rtol=.03)
    cases+=1
    # Legacy block128 dense selection regression: select every KV block.
    q=torch.randn(512,4,128,device='cuda',dtype=torch.bfloat16);k=torch.randn(512,1,128,device='cuda',dtype=q.dtype);v=torch.randn_like(k)
    cu=torch.tensor([0,512],device='cuda',dtype=torch.int32)
    out=lightning_moba.lightning_moba_topk_varlen_func(q,k,v,cu,cu,512,512,128,4,causal=True)
    expected=torch.nn.functional.scaled_dot_product_attention(q.transpose(0,1)[None],k.transpose(0,1)[None],v.transpose(0,1)[None],is_causal=True,enable_gqa=True)[0].transpose(0,1)
    torch.testing.assert_close(out.float(),expected.float(),atol=.035,rtol=.03)
    print(f'{cases} block8 package cases + {len(invalid)} invalid-input checks + legacy block128 regression passed',flush=True)

@pytest.mark.cuda
@pytest.mark.sm90
def test_block8_migration():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9,0):
        pytest.skip('block8 integration requires an SM90 GPU')
    main()


if __name__=='__main__':main()
