#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <math.h>

__global__ void add(float *A, float *B, float *C, int N)
{
    int thread_index = blockDim.x * blockIdx.x + threadIdx.x;

    if (thread_index < N)
        C[thread_index] = A[thread_index] + B[thread_index];
}

__global__ void singleThreadVecAdd(float *A, float *B, float *C, int N)
{
    for (int i = 0; i < N; ++i)
        C[i] = A[i] + B[i];
}

void generate_floats(float *A, int N)
{
    for (int i = 0; i < N; ++i)
        A[i] = sin(i) + cos(i);
}

int main()
{
    printf("\n\nProgram to perform Vector Addition in CUDA\n\n");
    int N = 2048; // Number of elements in the array

    float *A, *B, *C;
    float host_A[N], host_B[N], host_C[N];

    // generate random floating numbers for input
    printf("\nGenerating %d floating-point numbers for the input arrays....\n", N);
    generate_floats(host_A, N);
    generate_floats(host_B, N);

    printf("\nAllocating memory on the GPU...\n\n");
    // allocate space on device
    cudaMalloc((void **)&A, N * sizeof(float));
    cudaMalloc((void **)&B, N * sizeof(float));
    cudaMalloc((void **)&C, N * sizeof(float));

    // memory transfer from host to device
    printf("\nTransferring data from host to device for computations...\n\n");

    cudaMemcpy(A, host_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B, host_B, N * sizeof(float), cudaMemcpyHostToDevice);

    // dimensions of thread block + kernel launch
    int blockDim = 1024;

    int gridDim = ceil((float)(N) / 1024);

    printf("\n\nCalling the kernel with %d Blocks and %d threads in each block\n", gridDim, blockDim);

    double t1 = clock();
    add<<<gridDim, blockDim>>>(A, B, C, N);
    cudaDeviceSynchronize();
    double t2 = clock();

    printf("\nTime taken to add %d elements = %lf\n\n", N, (t2 - t1) / CLOCKS_PER_SECOND);

    // copy back to host
    printf("\n\nCalculation completed on the GPU. Fetching the answer back from the GPU's global memory\n");
    cudaMemcpy(host_C, C, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Calculating the time required for a single thread, within a single block
    t1 = clock();
    singleThreadVecAdd<<<1, 1>>>(A, B, C, N);
    cudaDeviceSynchronize();
    t2 = clock();

    // copy back to host
    cudaMemcpy(host_C, C, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\nTime taken to perform %d additions with single thread and One block: %lf\n", N, (t2 - t1) / CLOCKS_PER_SECOND);

    // free the malloc'ed memory
    printf("\n\nFree'ing the malloc'ed memory on the GPU\n");
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}
