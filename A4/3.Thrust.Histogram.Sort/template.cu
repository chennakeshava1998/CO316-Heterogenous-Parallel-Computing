#include <wb.h>

#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>

int main(int argc, char *argv[]) {
  wbArg_t args;
  int inputLength, num_bins;
  unsigned int *hostInput, *hostBins;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (unsigned int *)wbImport(wbArg_getInputFile(args, 0),
                                       &inputLength, "Integer");
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);

  // Creating a vector to be transferred onto the device
  thrust::device_vector<unsigned int> deviceInput(inputLength);
  // Copy the input to the GPU
  wbTime_start(GPU, "Allocating GPU memory");
  thrust::copy(hostInput, hostInput + deviceInput.size(), deviceInput.begin());

  //@@ Insert code here
  wbTime_stop(GPU, "Allocating GPU memory");

  // sorting on the device
  // sorting needs to be done, before counting the number of bins.
  thrust::sort(deviceInput.begin(), deviceInput.end());

  // Determine the number of bins (num_bins) and create space on the host
  //@@ insert code here
  num_bins = deviceInput.back() + 1;
  hostBins = (unsigned int *)malloc(num_bins * sizeof(unsigned int));

  // Allocate a device vector for the appropriate number of bins
  //@@ insert code here

  thrust::device_vector<unsigned int> deviceBins(num_bins);

  // Create a cumulative histogram. Use thrust::counting_iterator and
  // thrust::upper_bound
  //@@ Insert code here

  thrust::counting_iterator<int> toSearch(0);
  thrust::device_vector<unsigned int> temp;
  thrust::upper_bound(deviceInput.begin(), deviceInput.end(), toSearch + num_bins, temp.begin(), thrust::less<int>());


  // Use thrust::adjacent_difference to turn the culumative histogram
  // into a histogram.
  //@@ insert code here.
  thrust::adjacent_difference(temp.begin(), temp.end(), deviceBins.begin());


  // Copy the histogram to the host
  //@@ insert code here

  thrust::copy(deviceBins.begin(), deviceBins.end(), hostBins);

  // Check the solution is correct
  wbSolution(args, hostBins, num_bins);

  // Free space on the host
  //@@ insert code here
  free(hostBins);

  return 0;
}
