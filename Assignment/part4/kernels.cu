#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include<cuda_runtime.h>


 __global__ void gpu_residual (float *residuals, float* block_res) {

	 extern __shared__ float sdata[];
	 unsigned int tid = threadIdx.x;
	 int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	 sdata[tid] = residuals[i];
	 __syncthreads();

	 for(unsigned int s=blockDim.x/2; s>0; s>>=1) {
		 if (tid < s) {
		 	sdata[tid] += sdata[tid + s];
	 		__syncthreads();
			}
		}

		if (tid == 0) {
				block_res[blockIdx.x] = sdata[tid];
		}
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
