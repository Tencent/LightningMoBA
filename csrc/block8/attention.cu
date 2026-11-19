#ifndef OPT_REUSE
#define OPT_REUSE 1
#endif
#ifndef OPT_ASYNC
#define OPT_ASYNC 1
#endif
#ifndef OPT_VECTOR
#define OPT_VECTOR 1
#endif
#ifndef OPT_MASK
#define OPT_MASK 1
#endif
#ifndef OPT_FASTEXP
#define OPT_FASTEXP 1
#endif
#ifndef OPT_SKIP
#define OPT_SKIP 1
#endif
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <stdint.h>
#include "block8_launch.h"
using namespace nvcuda;
using bf=__nv_bfloat16;
__device__ inline void copy16_async(void* dst,const void* src){
 unsigned address=(unsigned)__cvta_generic_to_shared(dst);
 asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"r"(address),"l"(src));
}
__device__ inline void atomic_max_float(float* address, float value) {
 int* bits = reinterpret_cast<int*>(address);
 int old = atomicCAS(bits, 0, 0);
 while (__int_as_float(old) < value) {
  const int assumed = old;
  old = atomicCAS(bits, assumed, __float_as_int(value));
  if (old == assumed) break;
 }
}
// Each warp owns 16 gathered queries. Selection block8, execution column256.
// FIND_MAX first computes the true masked row maximum across all CSC columns.
template<int WARPS, bool FIND_MAX>
__global__ void attend(const bf* q,const bf* k,const bf* v,const uint32_t* keep,float* shift,
 const int64_t* offsets,const int* nnz,const int* indices,const int* work_slot,const int* work_tile,
 float* out,float* den,int* count,int S,int K,int H,int HK,int C,int W){
 constexpr int R=16*WARPS,D=128,N=128,LD=136;
 extern __shared__ __align__(32) unsigned char storage[];
 bf* sq=(bf*)storage;
 bf* sk=sq+R*LD;
 bf* weights=sk+N*LD;
 float* scores=(float*)(weights+R*LD);
 int tid=threadIdx.x,warp=tid/32,lane=tid%32;
 int item=blockIdx.x,slot=work_slot[item],tile=work_tile[item];
 int head=slot/C,col=slot%C,kh=head/(H/HK),len=nnz[slot];
 int entry=tile*R+warp*16+lane/2;
 int row=entry<len?indices[offsets[slot]+entry]:0;
 int64_t start=offsets[slot];
#if OPT_VECTOR
 for(int x=tid*8;x<R*D;x+=32*WARPS*8){
  int entry=tile*R+x/D;
  int row=entry<len?indices[start+entry]:0;
  uint4 val=entry<len?*reinterpret_cast<const uint4*>(q+((int64_t)head*S+row)*D+x%D):make_uint4(0,0,0,0);
  *reinterpret_cast<uint4*>(sq+(x/D)*LD+x%D)=val;
 }
#else
 for(int x=tid;x<R*D;x+=32*WARPS){
  int r=x/D,entry=tile*R+r;
  int row=entry<len?indices[start+entry]:0;
  sq[(x/D)*LD+x%D]=entry<len?q[((int64_t)head*S+row)*D+x%D]:__float2bfloat16(0);
 }
#endif
 uint32_t row_mask=entry<len?keep[((int64_t)head*S+row)*W+col]:0;
 wmma::fragment<wmma::accumulator,16,16,16,float> output[8];
 for(int j=0;j<8;j++)wmma::fill_fragment(output[j],0.f);
 int r0=warp*16+lane/4,r1=r0+8;
 int e0=tile*R+r0,e1=tile*R+r1;
 int qr0=e0<len?indices[start+e0]:0,qr1=e1<len?indices[start+e1]:0;
 uint32_t mask0=e0<len?keep[((int64_t)head*S+qr0)*W+col]:0;
 uint32_t mask1=e1<len?keep[((int64_t)head*S+qr1)*W+col]:0;
 float sh0=0,sh1=0;
 if constexpr (!FIND_MAX) {
  sh0=e0<len?shift[(int64_t)head*S+qr0]:0;
  sh1=e1<len?shift[(int64_t)head*S+qr1]:0;
 }
 float max0=-CUDART_INF_F,max1=-CUDART_INF_F;
 float d0=0,d1=0;int c0=0,c1=0;
 __syncthreads();
 for(int inner=0;inner<2;inner++){
  int base=col*256+inner*N;
#if OPT_SKIP
  if(!__syncthreads_or(((row_mask>>(inner*16))&0xffffu)!=0))continue;
#endif
#if OPT_VECTOR
#if OPT_ASYNC
  for(int x=tid*8;x<N*D;x+=32*WARPS*8)
   copy16_async(sk+(x/D)*LD+x%D,k+((int64_t)kh*K+base+x/D)*D+x%D);
  asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;"::);
#else
  for(int x=tid*8;x<N*D;x+=32*WARPS*8)
   *reinterpret_cast<uint4*>(sk+(x/D)*LD+x%D)=*reinterpret_cast<const uint4*>(k+((int64_t)kh*K+base+x/D)*D+x%D);
#endif
#else
  for(int x=tid;x<N*D;x+=32*WARPS)sk[(x/D)*LD+x%D]=k[((int64_t)kh*K+base+x/D)*D+x%D];
#endif
  __syncthreads();
  wmma::fragment<wmma::accumulator,16,16,16,float> accs[8];
  #pragma unroll
  for(int j=0;j<8;j++)wmma::fill_fragment(accs[j],0.f);
  #pragma unroll
  for(int d=0;d<D;d+=16){
   wmma::fragment<wmma::matrix_a,16,16,16,bf,wmma::row_major> a;
   wmma::load_matrix_sync(a,sq+warp*16*LD+d,LD);
   #pragma unroll
   for(int j=0;j<8;j++){
    wmma::fragment<wmma::matrix_b,16,16,16,bf,wmma::col_major> b;
    wmma::load_matrix_sync(b,sk+j*16*LD+d,LD);
    wmma::mma_sync(accs[j],a,b,accs[j]);
   }
  }
  #pragma unroll
  for(int j=0;j<8;j++){
   #pragma unroll
   for(int i=0;i<8;i++){
    bool high=(i&2)!=0;
    int n=j*16+(lane%4)*2+((i&4)?8:0)+(i&1);
    int token=base+n,block=token/8;
    bool live=high?e1<len:e0<len;
    int qr=high?qr1:qr0;
    uint32_t bits=high?mask1:mask0;
    bool allowed=live && token<=qr+K-S && ((bits>>(block%32))&1);
    // Round identically in both passes; do not fuse scale/subtract after the
    // maximum was rounded and stored, particularly for large finite logits.
    float score=__fmul_rn(accs[j].x[i],0.08838834764831845f);
    if constexpr (FIND_MAX) {
     if(allowed){if(high)max1=fmaxf(max1,score);else max0=fmaxf(max0,score);}
    } else {
     float value=allowed?__expf(__fsub_rn(score,high?sh1:sh0)):0;
     weights[(high?r1:r0)*LD+n]=__float2bfloat16_rn(value);
     if(high){d1+=value;c1+=allowed;}else{d0+=value;c0+=allowed;}
    }
   }
  }
  __syncthreads();
  if constexpr (FIND_MAX) continue;
#if OPT_VECTOR
#if OPT_ASYNC
  for(int x=tid*8;x<N*D;x+=32*WARPS*8)
   copy16_async(sk+(x/D)*LD+x%D,v+((int64_t)kh*K+base+x/D)*D+x%D);
  asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;"::);
#else
  for(int x=tid*8;x<N*D;x+=32*WARPS*8)
   *reinterpret_cast<uint4*>(sk+(x/D)*LD+x%D)=*reinterpret_cast<const uint4*>(v+((int64_t)kh*K+base+x/D)*D+x%D);
#endif
#else
  for(int x=tid;x<N*D;x+=32*WARPS)sk[(x/D)*LD+x%D]=v[((int64_t)kh*K+base+x/D)*D+x%D];
#endif
  __syncthreads();
#if OPT_REUSE
  #pragma unroll
  for(int n=0;n<N;n+=16){
   wmma::fragment<wmma::matrix_a,16,16,16,bf,wmma::row_major> a;
   wmma::load_matrix_sync(a,weights+warp*16*LD+n,LD);
   #pragma unroll
   for(int j=0;j<8;j++){
    wmma::fragment<wmma::matrix_b,16,16,16,bf,wmma::row_major> b;
    wmma::load_matrix_sync(b,sk+n*LD+j*16,LD);
    wmma::mma_sync(output[j],a,b,output[j]);
   }
  }
#else
  for(int j=0;j<8;j++)for(int n=0;n<N;n+=16){
   wmma::fragment<wmma::matrix_a,16,16,16,bf,wmma::row_major> a;
   wmma::fragment<wmma::matrix_b,16,16,16,bf,wmma::row_major> b;
   wmma::load_matrix_sync(a,weights+warp*16*LD+n,LD);
   wmma::load_matrix_sync(b,sk+n*LD+j*16,LD);
   wmma::mma_sync(output[j],a,b,output[j]);
  }
#endif
  __syncthreads();
 }
 if constexpr (FIND_MAX) {
  for(int mask=1;mask<=2;mask*=2){
   max0=fmaxf(max0,__shfl_xor_sync(0xffffffff,max0,mask));
   max1=fmaxf(max1,__shfl_xor_sync(0xffffffff,max1,mask));
  }
  if(lane%4==0){
   if(e0<len)atomic_max_float(shift+(int64_t)head*S+qr0,max0);
   if(e1<len)atomic_max_float(shift+(int64_t)head*S+qr1,max1);
  }
  return;
 }
 #pragma unroll
 for(int j=0;j<8;j++){
  wmma::store_matrix_sync(scores+warp*16*16,output[j],16,wmma::mem_row_major);
  __syncwarp();
  for(int x=lane;x<16*16;x+=32){
   int ent=tile*R+warp*16+x/16;
   if(ent<len){int qr=indices[start+ent];atomicAdd(out+((int64_t)head*S+qr)*D+j*16+x%16,scores[warp*16*16+x]);}
  }
  __syncwarp();
 }
 for(int mask=1;mask<=2;mask*=2){d0+=__shfl_xor_sync(0xffffffff,d0,mask);d1+=__shfl_xor_sync(0xffffffff,d1,mask);c0+=__shfl_xor_sync(0xffffffff,c0,mask);c1+=__shfl_xor_sync(0xffffffff,c1,mask);}
 if(lane%4==0){
  if(e0<len){atomicAdd(den+(int64_t)head*S+qr0,d0);atomicAdd(count+(int64_t)head*S+qr0,c0);}
  if(e1<len){atomicAdd(den+(int64_t)head*S+qr1,d1);atomicAdd(count+(int64_t)head*S+qr1,c1);}
 }

}
extern "C" int lightning_moba_block8_launch(const void*q,const void*k,const void*v,const void*keep,void*shift,
 const void*offsets,const void*nnz,const void*indices,const void*slots,const void*tiles,
 void*out,void*den,void*count,int S,int K,int H,int HK,int C,int W,int items,int warps,int device,void*stream){
 cudaError_t err=cudaSetDevice(device);if(err!=cudaSuccess)return (int)err;
 #define ARGS (const bf*)q,(const bf*)k,(const bf*)v,(const uint32_t*)keep,(float*)shift,(const int64_t*)offsets,(const int*)nnz,(const int*)indices,(const int*)slots,(const int*)tiles,(float*)out,(float*)den,(int*)count,S,K,H,HK,C,W
 #define RUN(NW) {constexpr int sm=(16*NW*136+128*136+16*NW*136)*2+16*NW*16*4; \
 constexpr int max_sm=(16*NW*136+128*136)*2; \
 err=cudaFuncSetAttribute(attend<NW,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,max_sm);if(err!=cudaSuccess)return (int)err; \
 err=cudaFuncSetAttribute(attend<NW,false>,cudaFuncAttributeMaxDynamicSharedMemorySize,sm);if(err!=cudaSuccess)return (int)err; \
 attend<NW,true><<<items,32*NW,max_sm,(cudaStream_t)stream>>>(ARGS); \
 err=cudaGetLastError();if(err!=cudaSuccess)return (int)err; \
 attend<NW,false><<<items,32*NW,sm,(cudaStream_t)stream>>>(ARGS);}
 if(warps==8)RUN(8) else if(warps==4)RUN(4) else if(warps==2)RUN(2) else if(warps==1)RUN(1) else return -1;
 return (int)cudaGetLastError();
}
