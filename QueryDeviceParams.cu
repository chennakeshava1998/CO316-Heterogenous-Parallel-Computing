// https://devblogs.nvidia.com/how-query-device-properties-and-handle-errors-cuda-cc/

// http://www.cs.fsu.edu/~xyuan/cda5125/examples/lect24/devicequery.cu

#include <stdio.h>
void printDeviceProp(cudaDeviceProp prop);

void printDeviceProp(cudaDeviceProp prop)
{
    printf("Major revision number:         %d\n", prop.major);
    printf("Minor revision number:         %d\n", prop.minor);
    printf("Name:                          %s\n", prop.name);
    printf("Total global memory:           %u\n", prop.totalGlobalMem);
    printf("Total shared memory per block: %u\n", prop.sharedMemPerBlock);
    printf("Total registers per block:     %d\n", prop.regsPerBlock);
    printf("Warp size:                     %d\n", prop.warpSize);
    printf("Maximum threads per block:     %d\n", prop.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of block:  %d\n", i, prop.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of grid:   %d\n", i, prop.maxGridSize[i]);

    printf("Clock rate:                    %d\n", prop.clockRate);
    printf("Total constant memory:         %u\n", prop.totalConstMem);
    printf("Number of multiprocessors:     %d\n", prop.multiProcessorCount);
}

int main()
{
    int n = 0;
    cudaGetDeviceCount(&n);

    printf("CUDA Device Properties:\n\n");
    for (int i = 0; i < n; ++i)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i); // Second argument is also imp.

        printf("Properties of Device - %d\n", i);

        printDeviceProp(prop);
        printf("\n\n");
    }

    return 0;
}
