#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <sys/time.h>

#define TILE_SIZE 16

using namespace std;


__global__ void init (float *A, float *x, float *y, int span1, int span2, int n)
{
	int i;
	int id = threadIdx.x + blockIdx.x * blockDim.x;

        for (i=span1*id; i<span1*(id+1); i++) {
                A[i] = (float)1/(i+1);
        }
	for (i=span2*id; i<span2*(id+1); i++) {
		x[i] = 1;
		y[i] = 0;
	}
}

__global__ void multiply (float *A, float *x, float *y, int span, int n)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	__shared__ float xs[TILE_SIZE];
	float local_sum = 0.0;

	for (int i=0; i<n/TILE_SIZE; ++i) {
		xs[threadIdx.x] = x[i*TILE_SIZE + threadIdx.x];
		__syncthreads();
		for (int j=0; j<TILE_SIZE; ++j) {
			local_sum += A[id*n+i*TILE_SIZE+j] * xs[j];
		}
		__syncthreads();
	}
	y[id] = local_sum;
}

int main(int argc, char *argv[]){
	float *A, *x, *y, *y_cpu;
	if (argc != 3) {
		printf("Need matrix dimension and thread count!\n");
	}
	int n = atoi(argv[1]), nthreads = atoi(argv[2]), num_blocks1 = -1, threads_per_block1 = -1, num_blocks2 = -1, threads_per_block2 = -1;
	struct timeval tv0, tv1, tv2, tv3;
	struct timezone tz0, tz1, tz2, tz3;
	// check nthreads is power of 2
	assert((nthreads & (nthreads - 1)) == 0);

	cudaMallocManaged((void**)&A, sizeof(float)*n*n);
	cudaMallocManaged((void**)&x, sizeof(float)*n);	
	cudaMallocManaged((void**)&y, sizeof(float)*n);
	y_cpu = (float*)malloc(n*sizeof(float));


	if (nthreads < 1024) {
		num_blocks2 = 1;
		threads_per_block2 = nthreads;
	} else {
		num_blocks2 = nthreads/1024;
		threads_per_block2 = 1024;
	}


	gettimeofday(&tv0, &tz0);

	init<<<num_blocks2, threads_per_block2>>>(A, x, y, (n*n)/nthreads, n/nthreads, n);
	cudaDeviceSynchronize();

	gettimeofday(&tv1, &tz1);

	multiply<<<nthreads/TILE_SIZE, TILE_SIZE>>>(A, x, y, (n*n)/nthreads, n);
	cudaDeviceSynchronize();

	gettimeofday(&tv2, &tz2);

	printf("Init time: %ld microseconds, Multiply time: %ld microseconds\n", (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec), (tv2.tv_sec-tv1.tv_sec)*1000000+(tv2.tv_usec-tv1.tv_usec));

	cudaError_t err = cudaGetLastError();        // Get error code

	if ( err != cudaSuccess ) {
		printf("CUDA Error: %s\n", cudaGetErrorString(err));
		exit(-1);
	}
	cudaDeviceSynchronize();

	err = cudaGetLastError();        // Get error code

        if ( err != cudaSuccess ) {
                printf("CUDA Error: %s\n", cudaGetErrorString(err));
                exit(-1);
        }

	for (int i=0; i<n; ++i) {
		y_cpu[i] = 0;
	}
/*
#pragma omp parallel for num_threads (nthreads)
	for (int i=0; i<n*n; ++i) {
		int r = i/n, c = i%n;
		float temp = A[i] * x[c];
	#pragma omp critical
		y_cpu[r] += temp;
	}
*/

	for (int i=0; i<n; ++i) {
		float sum = 0.0;
		for (int j=0; j<n; ++j) {
			sum += A[i*n+j] * x[j];
		}
		y_cpu[i] = sum;
	}

	gettimeofday(&tv3, &tz3);

	printf("CPU time: %ld microseconds\n", (tv3.tv_sec-tv2.tv_sec)*1000000+(tv3.tv_usec-tv2.tv_usec));

	float diff = 0.0;
	for (int i=0; i<n; ++i) {
		diff += abs(y[i] - y_cpu[i]);
	}
	printf("Diff sum: %.20f, y = %.20f\n",diff, y[random()%n]);
//	for (int i=0; i<n; ++i) {
//		printf("%f, ", A[i]);
//	}
	printf("\n");
	return 0;
}
