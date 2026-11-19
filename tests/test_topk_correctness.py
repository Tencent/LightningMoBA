"""Selection correctness across multiple KM tiles and repeated shared-memory reuse."""
import pytest
import torch
import lightning_moba_cuda as native

pytestmark = pytest.mark.cuda

@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
@pytest.mark.parametrize('dim',[64,128,256])
@pytest.mark.parametrize('causal',[False,True])
@pytest.mark.parametrize('topk',[16,32,64])
def test_native_topk(dtype,dim,causal,topk):
    if not torch.cuda.is_available():
        pytest.skip('TopK selection requires a CUDA GPU')
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip('TopK selection requires Ampere or newer')
    torch.manual_seed(123)
    n,h,c=4608,4,64
    q=torch.randn(n,h,dim,device='cuda',dtype=dtype)
    km=torch.randn(n//c,1,dim,device='cuda',dtype=dtype)
    cu=torch.tensor([0,n],device='cuda',dtype=torch.int32)
    cum=torch.tensor([0,n//c],device='cuda',dtype=torch.int32)
    # FP64 avoids allowing the process-wide TF32 override to affect the reference.
    scores=torch.einsum('qhd,chd->qhc',q.double(),km.expand(-1,h,-1).double())
    for _ in range(3):
        offsets,nnz,flat,values,ids=native.moba_fused_topk(q,km,cu,cu,cum,n,n,topk,c,causal)
        valid=ids>=0
        current=torch.arange(n,device='cuda')//c
        expected=torch.clamp(current+1,max=topk) if causal else torch.full_like(current,topk)
        torch.testing.assert_close(valid.sum(-1),expected[:,None].expand(n,h))
        assert bool((ids<n//c).all())
        ordered=ids.sort(-1).values
        assert not bool(((ordered[...,1:]==ordered[...,:-1])&(ordered[...,1:]>=0)).any())
        assert int(nnz.sum())==int(valid.sum())
        if causal:
            assert bool((ids<=current[:,None,None]).all())
            for anchor in [torch.zeros_like(current),current,torch.clamp(current-1,min=0)]:
                assert bool((ids==anchor[:,None,None]).any(-1).all())
        chosen=torch.zeros_like(scores,dtype=torch.int32)
        chosen.scatter_add_(-1,ids.clamp_min(0).long(),valid.int())
        eligible=torch.ones_like(scores,dtype=torch.bool)
        cols=torch.arange(n//c,device='cuda')[None,None,:]
        if causal:
            eligible &= cols<current[:,None,None]
            eligible &= cols!=0
            eligible &= cols!=torch.clamp(current-1,min=0)[:,None,None]
        worst_selected=scores.masked_fill(~((chosen>0)&eligible),float('inf')).amin(-1)
        best_omitted=scores.masked_fill(~((chosen==0)&eligible),-float('inf')).amax(-1)
        assert bool((worst_selected+.05>=best_omitted).all())
        scored=valid & values.isfinite()
        torch.testing.assert_close(values[scored].double(),scores.gather(-1,ids.clamp_min(0).long())[scored],atol=.005,rtol=.0001)
