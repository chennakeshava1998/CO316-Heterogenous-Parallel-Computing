#include <stdio.h>
#include <math.h>

#define M 1024
#define N 4096

__global__ void add(float A[M][N], float B[M][N], float C[M][N])
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col < N && row < M)
        C[row * N + col] = A[row * N + col] + B[row * N + col];
}

__global__ void singleThreadVecAdd(float A[M][N], float B[M][N], float C[M][N])
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
            C[i][j] = A[i][j] + B[i][j];
    }
}

int main()
{
    printf("\n\nProgram to perform Vector Addition in CUDA\n\n");

    float *A, *B, *C;
    float host_A[M][N], host_B[M][N], host_C[M][N];

    // generate random floating numbers for input
    printf("\nGenerating %d floating-point numbers for the input arrays....\n", N * M);

    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
            host_A[i * N + j] = sin(i) + sin(j);
    }

    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
            host_B[i * N + j] = cos(i) + cos(j);
    }

    printf("\nAllocating memory on the GPU...\n\n");
    // allocate space on device
    cudaMalloc((void **)&A, M * N * sizeof(float));
    cudaMalloc((void **)&B, M * N * sizeof(float));
    cudaMalloc((void **)&C, M * N * sizeof(float));

    // memory transfer from host to device
    printf("\nTransferring data from host to device for computations...\n\n");

    cudaMemcpy(A, host_A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B, host_B, M * N * sizeof(float), cudaMemcpyHostToDevice);

    // dimensions of thread block + kernel launch
    int blockDim = 1024;

    dim3 gridDim(ceil((float)(M) / 1024), ceil((float)(N) / 1024), 1);

    printf("\n\nCalling the kernel with %d Blocks and %d threads in each block\n", gridDim, blockDim);

    // timing the GPU kernel
    double t1 = myCPUTimer();

    add<<<gridDim, blockDim>>>(A, B, C);
    cudaDeviceSynchronize();
    double t2 = myCPUTimer();

    printf("\nNumber of threads per block: %d\n", blockDim);
    printf("\nDimesions of the grid: %d BY %d BY %d\n", gridDim.x, gridDim.y, gridDim.z);
    printf("\nTime taken to add %d elements = %lf\n\n", M * N, t2 - t1);

    // copy back to host
    printf("\n\nCalculation completed on the GPU. Fetching the answer back from the GPU's global memory\n");
    cudaMemcpy(host_C, C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Calculating the time required for a single thread, within a single block
    t1 = myCPUTimer();
    singleThreadVecAdd<<<1, 1>>>(A, B, C);
    cudaDeviceSynchronize();
    t2 = myCPUTimer();

    // copy back to host
    cudaMemcpy(host_C, C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\nTime taken to perform %d additions with single thread and One block: %lf\n", M * N, t2 - t1);

    // free the malloc'ed memory
    printf("\n\nFree'ing the malloc'ed memory on the GPU\n");
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}
