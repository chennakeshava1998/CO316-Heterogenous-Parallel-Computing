#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <math.h>

#define M 512
#define N 512

__global__ void add(int *A, int *B, int *C)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col < N && row < M)
        C[row * N + col] = A[row * N + col] + B[row * N + col];
}

__global__ void singleThreadVecAdd(int *A, int *B, int *C)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
            C[i * N + j] = A[i * N + j] + B[i * N + j];
    }
}

int main()
{
    printf("\n\nProgram to perform Vector Addition in CUDA\n\n");

    int *A, *B, *C;
    int host_A[M][N], host_B[M][N], host_C[M][N];

    // generate random int numbers for input
    printf("\nGenerating %d int numbers for the input arrays....\n", N * M);
    int i,j;
    for (i = 0; i < M; ++i)
    {
        for (j = 0; j < N; ++j)
            host_A[i][j] = sin(i) + sin(j);
//            host_A[i][j] = 1.0;


    }

    for (i = 0; i < M; ++i)
    {
        for (j = 0; j < N; ++j)
            host_B[i][j] = 1.0;
    }

    printf("\nAllocating memory on the GPU...\n\n");
    // allocate space on device
    cudaMalloc((void **)&A, M * N * sizeof(int));
    cudaMalloc((void **)&B, M * N * sizeof(int));
    cudaMalloc((void **)&C, M * N * sizeof(int));

    // memory transfer from host to device
    printf("\nTransferring data from host to device for computations...\n\n");

    cudaMemcpy(A, host_A, M * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(B, host_B, M * N * sizeof(int), cudaMemcpyHostToDevice);

    // dimensions of thread block + kernel launch
    dim3 blockDim(16, 16, 1);

    dim3 gridDim((int)ceil((float)(M) / blockDim.x),(float) ceil((int)(N) / blockDim.y), 1);

    printf("\n\nCalling the kernel with %d Blocks and %d threads in each block\n", gridDim, blockDim);

    // timing the GPU kernel
    double t1 = clock();

    add<<<gridDim, blockDim>>>(A, B, C);
    cudaDeviceSynchronize();
    double t2 = clock();

    printf("\nNumber of threads per block: %d\n", blockDim.x * blockDim.y);
    printf("\nDimesions of the grid: %d BY %d BY %d\n", gridDim.x, gridDim.y, gridDim.z);
    printf("\nTime taken to add %d elements = %lf\n\n", M * N, (t2 - t1) / CLOCKS_PER_SEC);

    // copy back to host
    printf("\n\nCalculation completed on the GPU. Fetching the answer back from the GPU's global memory\n");
    cudaMemcpy(host_C, C, M * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Calculating the time required for a single thread, within a single block
    t1 = clock();
    singleThreadVecAdd<<<1, 1>>>(A, B, C);
    cudaDeviceSynchronize();
    t2 = clock();

    // copy back to host
    cudaMemcpy(host_C, C, M * N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nTime taken to perform %d additions with single thread and One block: %lf\n", M * N, (t2 - t1) / CLOCKS_PER_SEC);

    // free the malloc'ed memory
    printf("\n\nFree'ing the malloc'ed memory on the GPU\n");
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}
