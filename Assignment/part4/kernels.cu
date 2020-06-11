#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include<cuda_runtime.h>


__device__ void warpReduce(volatile float* sdata, int tid) {
	sdata[tid] += sdata[tid + 32];
	sdata[tid] += sdata[tid + 16];
	sdata[tid] += sdata[tid + 8];
	sdata[tid] += sdata[tid + 4];
	sdata[tid] += sdata[tid + 2];
	sdata[tid] += sdata[tid + 1];
}



	// Slide 22 : http://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
 __global__ void gpu_residual (float* residual, float *u, float* u_help, int Dim) {
 	extern __shared__ float sdata[];

	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int i = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	
	if(i>Dim) return;
	
	float diff = u_help[i] - u[i];
 	sdata[i] = diff * diff; 
 	__syncthreads();

 	for(unsigned int s=Dim/2; s>0; s>>=1) {
 		if (i > s) return;
 		sdata[i] += sdata[i + s];
 		__syncthreads();
 	}

 	//if (i < 32) warpReduce(sdata, i);
 	if (i == 0) residual[0] = sdata[0];
}


__global__ void gpu_Heat (float *h, float *g, int N) {
  // global thread IDs
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  int j = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (i>0 && j>0 && i < N-1 && j < N-1) {
    g[i*N+j]= 0.25f * (h[ i*N     + (j-1) ]+  // left
					            h[ i*N     + (j+1) ]+  // right
				              h[ (i-1)*N + j     ]+  // top
				              h[ (i+1)*N + j     ]); // bottom
  }
}