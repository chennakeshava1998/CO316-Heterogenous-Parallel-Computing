#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define Mask_width 5
#define Mask_radius Mask_width / 2
#define TILE_WIDTH 16
#define THREADS 16
#define w (TILE_WIDTH + Mask_width - 1)
#define clamp(x) (min(max((x), 0.0), 1.0))

//@@ INSERT CODE HERE
void convolution(float * deviceInputImageData, const float * __restrict__ deviceMaskData,unsigned char *deviceOutputImageData,int imageChannels,int imageWidth,int imageHeight)
{
  int i = threadIdx.y + blockDim.y * blockIdx.y;
  int j = threadIdx.x + blockDim.x * blockIdx.x;

  if(i<imageHeight && j < imageWidth)
  {
    for(int k = 0;k < imageChannels; ++k)
    {
      double accum = 0;

      for(int x = -Mask_radius; x<=Mask_radius;++x)
      {
        for(int y = -Mask_radius; y<=Mask_radius;++y)
        {
          xOffset = j + x;
          yOffset = i + y;
          if(xOffset >= 0 && xOffset < width && yOffset >= 0 && yOffset < height)
          { 
            imagePixel = deviceInputImageData[(yOffset * imageWidth + xOffset) * imageChannels + k];

            maskValue = deviceMaskData[(y+Mask_radius)*Mask_width+x+Mask_radius];
            accum += imagePixel * maskValue;

          }
        }
      }

      deviceOutputImageData[(i * imageWidth + j)*imageChannels + k] = clamp(accum, 0, 1)
    }
  }
}

int main(int argc, char *argv[]) {
  wbArg_t arg;
  int maskRows;
  int maskColumns;
  int imageChannels;
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  char *inputMaskFile;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  float *hostMaskData;
  float *deviceInputImageData;
  float *deviceOutputImageData;
  float *deviceMaskData;

  arg = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(arg, 0);
  inputMaskFile  = wbArg_getInputFile(arg, 1);

  inputImage   = wbImport(inputImageFile);
  hostMaskData = (float *)wbImport(inputMaskFile, &maskRows, &maskColumns);

  assert(maskRows == 5);    /* mask height is fixed to 5 in this mp */
  assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

  imageWidth    = wbImage_getWidth(inputImage);
  imageHeight   = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);

  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

  hostInputImageData  = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ INSERT CODE HERE
  wbCheck(cudaMalloc((void **)&deviceInputImageData,imageSize*sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceMaskData,5*5*sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutputImageData,imageSize*sizeof(unsigned char)));
  wbTime_stop(GPU, "Doing GPU memory allocation");
  
  wbTime_start(Copy, "Copying data to the GPU");
  //@@ INSERT CODE HERE
  cudaMemcpy(deviceInputImageData, hostInputImageData
    imageWidth * imageHeight * imageChannels * sizeof(float),
    cudaMemcpyHostToDevice);

  cudaMemcpy(deviceMaskData, hostMaskData
      5*5*sizeof(float)),
      cudaMemcpyHostToDevice);


  wbTime_stop(Copy, "Copying data to the GPU");

  dim3 dimBlock(THREADS, THREADS, 1);
  dim3 dimGrid((imageHeight - 1)/dimBlock.x + 1, (imageWidth - 1)/dimBlock.y + 1, 1);


  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ INSERT CODE HERE
  convolution<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceMaskData,
                                     deviceOutputImageData, imageChannels,
                                     imageWidth, imageHeight);
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ INSERT CODE HERE
  cudaMemcpy(hostOutputImageData, deviceOutputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float),
             cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  wbSolution(arg, outputImage);

  //@@ Insert code here

  free(hostMaskData);
  free(deviceInputImageData);
  free(deviceMaskData);
  free(deviceOutputImageData);

  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}
