#include <math.h>
#include <float.h>
#include <cuda.h>

__global__ void gpu_Heat (float *h, float *g, int N) {

	// TODO: kernel computation

  // global thread IDs
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  int j = (blockIdx.y * blockDim.y) + threadIdx.y;

  // check if thread is responsible for first column/row
  bool first_row = threadIdx.y + blockIdx.y == 0
  bool first_col = threadIdx.x + blockIdx.x == 0

  if (i < N-1 && j < N-1 && !first_row && !first_col) {
    g[i*N+j]= 0.25 * (h[ i*N     + (j-1) ]+  // left
					            h[ i*N     + (j+1) ]+  // right
				              h[ (i-1)*N + j     ]+  // top
				              h[ (i+1)*N + j     ]); // bottom
  }
}

// // Slide 22 : http://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
// __global__ void gpu_reduction (float *g_idata, float *g_odata) {
// 	extern __shared__ float sdata[];
//
// 	unsigned int tid = threadIdx.x;
// 	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
// 	sdata[tid] = g_idata[i];
// 	__syncthreads();
//
// 	for(unsigned int s=blockDim.x/2; s>32; s>>=1) {
// 		if (tid < s)
// 			sdata[tid] += sdata[tid + s];
// 		__syncthreads();
// 	}
//
// 	if (tid < 32) warpReduce(sdata, tid);
// 	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
// }
