#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <sys/time.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define MAX_ITR 1000
#define TOL 1e-5

__managed__ float diff = 0.0;
__managed__ int Itr = 0;
__managed__ bool done = 0;

__global__ void init (float *A, int span, int n)
{
	int i;
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	curandState_t state;
	curand_init(id, 0, 0, &state);
        for (i=span*id; i<span*(id+1); i++) {
		int r = i/n, c = i%n;
                A[(r+1)*(n+2)+c+1] = ((float)(curand(&state) % 100)/100.0);
        }
}

__global__ void solver (float *A, int span, int n)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int start = id * span, end = start + span - 1, itr = 0;
	float local_diff = 0.0, temp;

	while (!done && itr < MAX_ITR) {
		local_diff = 0.0;
		for (int idx=start; idx<=end; ++idx) {
			int i = idx/n, j = idx%n;
			i++; j++;
			temp = A[i*(n+2)+j];
			A[i*(n+2)+j] = 0.2 * (A[i*(n+2)+j] + A[(i-1)*(n+2)+j] + A[(i+1)*(n+2)+j] + A[i*(n+2)+(j-1)] + A[i*(n+2)+(j+1)]);
//			A[i*(n+2)+j] = 0.2 * (A[i*(n+2)+j]);
			local_diff += abs(A[i*(n+2)+j] - temp);
		}
		atomicAdd(&diff, local_diff);
//		cg::grid_group grid = cg::this_grid();
//		grid.sync();
		__syncthreads();
		itr++;
		if (id == 0) {
			if ((float)diff/(n*n) < TOL) {
				done = 1;
			}
			printf("[itr]: %d, [diff]: %f\n", itr, (float)diff/(n*n));
			diff = 0.0;
		}
		__syncthreads();
//		grid.sync();

	}

	if (id == 0) {
		Itr = itr;
	}

}

int main(int argc, char *argv[]){
	float *A;
	if (argc != 3) {
		printf("Need matrix dimension and thread count!\n");
	}
	int n = atoi(argv[1]), nthreads = atoi(argv[2]), num_blocks1 = -1, threads_per_block1 = -1, num_blocks2 = -1, threads_per_block2 = -1;
	struct timeval tv0, tv1, tv2;
	struct timezone tz0, tz1, tz2;
	// check nthreads is power of 2
	assert((nthreads & (nthreads - 1)) == 0);

	cudaMallocManaged((void**)&A, sizeof(float)*(n+2)*(n+2));

	if (nthreads < 32) {
		num_blocks1 = 1;
		threads_per_block1 = nthreads;
	} else {
		num_blocks1 = nthreads/32;
		threads_per_block1 = 32;
	}

	if (nthreads < 1024) {
		num_blocks2 = 1;
		threads_per_block2 = nthreads;
	} else {
		num_blocks2 = nthreads/1024;
		threads_per_block2 = 1024;
	}


	gettimeofday(&tv0, &tz0);

	init<<<num_blocks2, threads_per_block2>>>(A, (n*n)/nthreads, n);
	cudaDeviceSynchronize();

	int i;
	for (i=0; i<n+2; ++i) {A[i] = 0;} 		//upper pad
	for (i=0; i<(n+2)*(n+2); i+=n+2) {A[i] = 0;}	//left pad
	for (i=n+1; i<(n+2)*(n+2); i+=n+2) {A[i] = 0;}	//right pad
	for (i=(n+1)*(n+2); i<(n+2)*(n+2); i++) {A[i] = 0;}	//bottom pad

	cudaError_t err = cudaGetLastError();        // Get error code

        if ( err != cudaSuccess ) {
                printf("CUDA Error [Init]: %s\n", cudaGetErrorString(err));
                exit(-1);
        }
        cudaDeviceSynchronize();

	gettimeofday(&tv1, &tz1);

	solver<<<1, nthreads>>>(A, (n*n)/nthreads, n);
	cudaDeviceSynchronize();

	gettimeofday(&tv2, &tz2);

	printf("Init time: %ld microseconds, Convergence time: %ld microseconds\n", (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec), (tv2.tv_sec-tv1.tv_sec)*1000000+(tv2.tv_usec-tv1.tv_usec));

	err = cudaGetLastError();        // Get error code

	if ( err != cudaSuccess ) {
		printf("CUDA Error [Solver]: %s\n", cudaGetErrorString(err));
		exit(-1);
	}
	cudaDeviceSynchronize();

	printf("Diff: %f, Itr: %d\n",diff, Itr);
	return 0;
}
