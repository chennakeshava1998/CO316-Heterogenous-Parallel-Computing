#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <math.h>

// Number of rows in the first input matrix
#define M 512

// Number of columns in the first input matrix
#define N 512

// Dimensions of the input matrices: (M, N) and (N, M)

__global__ void matMul(int *A, int *B, int *C)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col < M && row < M)
    {
        long long prod_val = 0;
        for (int k = 0; k < N; ++k)
            prod_val += (A[row * N + k] * B[k * N + col]);

        C[row * N + col] = prod_val;
    }
}

__global__ void singleThreadVecMul(int *A, int *B, int *C)
{
    for (int row = 0; row < M; ++row)
    {
        for (int col = 0; col < M; ++col)
        {
            long long prod_val = 0;
            for (int k = 0; k < N; ++k)
                prod_val += (A[row * N + k] * B[k * N + col]);

            C[row * N + col] = prod_val;
        }
    }
}

void CPUMatMul(int A[M][N], int B[N][M], int C[M][M])
{

    for (int row = 0; row < M; ++row)
    {
        for (int col = 0; col < M; ++col)
        {
            int prod_val = 0;
            for (int k = 0; k < N; ++k)
	    {
            	prod_val = prod_val + (A[row][k] * B[k][col]);
	    }
            C[row][col] = prod_val;
        }
    }

    
}

bool compare(int A[M][M], int B[M][M], double accuracy)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < M; ++j)
            if ((abs(A[i][j] - B[i][j])) > accuracy)
                return 0;
    }

    return 1;
}

int main()
{
    printf("\n\nProgram to perform Matrix Multiplication in CUDA\n\n");

    int *A, *B, *C;
    int host_A[M][N], host_B[N][M], host_C[M][M], CPUMatMulAns[M][M];

    // generate random int numbers for input
    printf("\nGenerating %d int numbers for the input arrays....\n", N * M);
    int i, j;
    for (i = 0; i < M; ++i)
    {
        for (j = 0; j < N; ++j)
            host_A[i][j] = sin(i) + sin(j);
    }

    for (i = 0; i < N; ++i)
    {
        for (j = 0; j < M; ++j)
            host_B[i][j] = cos(i) + cos(j);
    }

    CPUMatMul(host_A, host_B, CPUMatMulAns);

    printf("\nAllocating memory on the GPU...\n\n");
    // allocate space on device
    cudaMalloc((void **)&A, M * N * sizeof(int));
    cudaMalloc((void **)&B, M * N * sizeof(int));
    cudaMalloc((void **)&C, M * M * sizeof(int));

    // memory transfer from host to device
    printf("\nTransferring data from host to device for computations...\n\n");

    cudaMemcpy(A, host_A, M * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(B, host_B, M * N * sizeof(int), cudaMemcpyHostToDevice);

    // dimensions of thread block + kernel launch
    dim3 blockDim(16, 16, 1);

    dim3 gridDim((int)ceil((float)(M) / blockDim.x), (float)ceil((int)(N) / blockDim.y), 1);

    printf("\n\nCalling the kernel with %d Blocks and %d threads in each block\n", gridDim.x * gridDim.y, blockDim.x * blockDim.y);

    // timing the GPU kernel
    double t1 = clock();

    matMul<<<gridDim, blockDim>>>(A, B, C);
    cudaDeviceSynchronize();
    double t2 = clock();

    // copy back to host
    printf("\n\nCalculation completed on the GPU. Fetching the answer back from the GPU's global memory\n");
    cudaMemcpy(host_C, C, M * M * sizeof(int), cudaMemcpyDeviceToHost);

    // checking of the required accuracy is attained
    double accuracy = pow(10, -6);
    if (compare(CPUMatMulAns, host_C, accuracy))
        printf("The answers generated by GPU are within %lf accuracy\n\n", accuracy);

    else
        printf("The answers generated by GPU are NOT within %lf accuracy\n\n", accuracy);

    printf("\nNumber of threads per block: %d\n", blockDim.x * blockDim.y);
    printf("\nDimesions of the grid: %d BY %d BY %d\n", gridDim.x, gridDim.y, gridDim.z);
    printf("\nTotal Time taken for %ld operations = %lf\n\n", M * M * N, (t2 - t1) / CLOCKS_PER_SEC);

    // Calculating the time required for a single thread, within a single block
    t1 = clock();
    singleThreadVecMul<<<1, 1>>>(A, B, C);
    cudaDeviceSynchronize();
    t2 = clock();

    // copy back to host
    cudaMemcpy(host_C, C, M * M * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nTime taken to perform %ld additions with single thread and One block: %lf\n", M * M * N, (t2 - t1) / CLOCKS_PER_SEC);

    // free the malloc'ed memory
    printf("\n\nFree'ing the malloc'ed memory on the GPU\n");
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}
