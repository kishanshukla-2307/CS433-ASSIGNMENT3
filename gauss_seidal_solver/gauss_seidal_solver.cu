#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <sys/time.h>

using namespace std;

__managed__ float diff = 0.0;
__managed__ int itr = 0;

__global__ void init (float *A, int span)
{
	int i;
	int id = threadIdx.x + blockIdx.x * blockDim.x;

        for (i=span*id; i<span*(id+1); i++) {
                A[i] = ((float)(random() % 100)/100.0);
        }
}

__global__ void solver (float *A, int span, int n)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int start = id * span, end = start + span - 1;
	float local_diff = 0.0, temp;
	while (!done) {
		for (int idx=start; idx<=end; ++idx) {
			int i = idx/n, j = idx%n;
			temp = A[idx];
			A[idx] = 0.2 * (A[idx] + A[(i-1)*n+j] + A[(i+1)*n+j] + A[i*n+(j-1)] + A[i*n+(j+1)]);
			local_diff += abs(A[idx] - temp);
		}
	}

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

	init<<<num_blocks2, threads_per_block2>>>(A, x, y, (n*n)/nthreads, n/nthreads, n);
	cudaDeviceSynchronize();

	gettimeofday(&tv1, &tz1);

	multiply<<<num_blocks1, threads_per_block1>>>(A, x, y, (n*n)/nthreads, n);
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

	printf("OpenMP time: %ld microseconds\n", (tv3.tv_sec-tv2.tv_sec)*1000000+(tv3.tv_usec-tv2.tv_usec));

	float diff = 0.0;
	for (int i=0; i<n; ++i) {
		diff += abs(y[i] - y_cpu[i]);
	}
	printf("Diff sum: %f, y = %f\n",diff, y_cpu[random()%n]);
//	for (int i=0; i<n; ++i) {
//		printf("%f, ", y_cpu[i]);
//	}
	printf("\n");
	return 0;
}
