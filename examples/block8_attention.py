"""Package-level block8 forward example; use --dense-check for dense selection."""
import argparse,math
import torch
from lightning_moba import lightning_moba_block8_func
p=argparse.ArgumentParser();p.add_argument('--backend',choices=['triton','cuda'],default='cuda');p.add_argument('--seq-len',type=int,default=513);p.add_argument('--dense-check',action='store_true');a=p.parse_args()
torch.manual_seed(42)
q=torch.randn(a.seq_len,4,128,device='cuda',dtype=torch.bfloat16)
k=torch.randn(a.seq_len,1,128,device='cuda',dtype=q.dtype);v=torch.randn_like(k)
alpha=1e-30 if a.dense_check else .1
out,counts=lightning_moba_block8_func(q,k,v,alpha,backend=a.backend,return_counts=True)
if a.dense_check:
    ref=torch.nn.functional.scaled_dot_product_attention(q.transpose(0,1)[None],k.transpose(0,1)[None],v.transpose(0,1)[None],is_causal=True,enable_gqa=True)[0].transpose(0,1)
    torch.testing.assert_close(out.float(),ref.float(),atol=.035,rtol=.03)
print({'shape':tuple(out.shape),'backend':a.backend,'mean_visible_tokens':float(counts.double().mean()),'dense_check':a.dense_check})
