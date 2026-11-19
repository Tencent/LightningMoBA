#include <cuda_runtime.h>
#include <mma.h>
#include "block8_launch.h"
using namespace nvcuda;
__global__ void block8_layout(float* out){
 wmma::fragment<wmma::accumulator,16,16,16,float> acc;
 for(int i=0;i<acc.num_elements;i++)acc.x[i]=threadIdx.x*100+i;
 wmma::store_matrix_sync(out,acc,16,wmma::mem_row_major);
}
extern "C" int lightning_moba_block8_probe(void* out,void* stream,int device){cudaError_t err=cudaSetDevice(device);if(err!=cudaSuccess)return (int)err;block8_layout<<<1,32,0,(cudaStream_t)stream>>>((float*)out);return cudaGetLastError();}
