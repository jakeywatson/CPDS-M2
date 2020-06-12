#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define sizex 16
#define sizey 16
#define sizesq sizey * sizex


__device__ void warpReduce(volatile float* sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}


__global__ void gpu_Heat (float *h, float *g, float *residuals, int N) {
  // global thread IDs
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  int j = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (i>0 && j>0 && i < N-1 && j < N-1) {
      g[i*N+j]= 0.25f * (h[ i*N     + (j-1) ]+  // left
					               h[ i*N     + (j+1) ]+  // right
				                 h[ (i-1)*N + j     ]+  // top
				                 h[ (i+1)*N + j     ]); // bottom
     float diff = g[(i*N)+j] - h[(i*N) + j];
		 residuals[(i*N)+j] = diff * diff;
  }
}

__global__ void gpu_residual (float *residuals, float* block_res) {

  extern __shared__ float sdata[];
  unsigned int tid = threadIdx.x;
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  sdata[tid] = residuals[i];
  __syncthreads();

  for(unsigned int s=blockDim.x/2; s>32; s>>=1) {
    if (tid < s) {
     sdata[tid] += sdata[tid + s];
     __syncthreads();
     }
   }

   if (tid < 32) warpReduce(sdata, tid);
   if (tid == 0) {
       block_res[blockIdx.x] = sdata[tid];
   }
}

// __global__ void gpu_Heat(float *h, float *g, float *block_res, int N) {
//
//      int tidx = threadIdx.x;
//      int tidy = threadIdx.y;
//      int bx = blockIdx.x * blockDim.x;
//      int by = blockIdx.y * blockDim.y;
//
//      __shared__ float sh_u[sizey+2][sizex+2];
//      extern __shared__ float res[];
//
//      #pragma unroll
//      for (int i = tidy; i<sizey+2; i+=sizey) {
//          #pragma unroll
//          for (int j = tidx; j<sizex+2; j+=sizex) {
//              int y = by+i-1;
//              int x = bx+j-1;
//              if (x>=0 && x<N && y>=0 && y<N) {
//                  sh_u[i][j] = h[x*N+y];
//                  res[i*N + j] = 0;
//              }
//          }
//      }
//      __syncthreads();
//
//      int x = bx+tidx;
//      int y = by+tidy;
//      if (x>0 && x<N-1 && y>0 && y<N-1) {
//          int i = tidx+1;
//          int j = tidy+1;
//          g[x*N+y] = 0.25f * (
//              sh_u[i+1][j] +
//              sh_u[i-1][j] +
//              sh_u[i][j-1] +
//              sh_u[i][j+1]);
//          float diff = g[(i*N)+j] - h[(i*N) + j];
//     		 res[i*N+j] = diff * diff;
//      }
//      __syncthreads();
//
//      int Dim = blockDim.x * blockDim.y;
//      int tid = tidy * blockDim.x + tidx;
//
//      for(unsigned int s=Dim/2; s>32; s>>=1) {
//   		 if (tid < s) {
//   		 	res[tid] += res[tid + s];
//   	 		__syncthreads();
//   			}
//   		}
//
//       if (tid < 32) warpReduce(res, tid);
//   		if (tid == 0) {
//   				block_res[blockIdx.x] = res[tid];
//   		}
// }
