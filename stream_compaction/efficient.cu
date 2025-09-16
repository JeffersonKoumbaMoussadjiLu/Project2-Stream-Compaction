#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		// __global__ kernels and device code here

		// Up-Sweep (reduce) kernel
        __global__ void kernUpSweep(int n, int step, int* data) {
			int index = (threadIdx.x + blockIdx.x * blockDim.x) * step + (step - 1); // get the index of the last element in the current segment

			// make sure we don't go out of bounds
            if (index < n) {
				data[index] += data[index - step / 2]; // add the value of the left child to the parent
            }
        }

		// Down-Sweep kernel
        __global__ void kernDownSweep(int n, int step, int* data) {

			int index = (threadIdx.x + blockIdx.x * blockDim.x) * step + (step - 1); // get the index of the last element in the current segment

			// make sure we don't go out of bounds
            if (index < n) {
				int left = index - step / 2; // get the index of the left child
				int t = data[left]; // store the left child's value
				data[left] = data[index]; // set the left child's value to the parent's value
				data[index] += t; // set the parent's value to the sum of the left child's value and the parent's value
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

			// TODO
            // Edge case: if n <= 0, return
            if (n <= 0) {
                return;
            }

            // Allocate space rounded up to next power of two
			int pow2 = 1 << ilog2ceil(n); // next power of 2
			int* dev_data = nullptr; // device array
			cudaMalloc(&dev_data, pow2 * sizeof(int)); // allocate device memory
			checkCUDAError("cudaMalloc dev_data for efficient scan"); // check for errors

            // Copy input data to device and pad the rest with 0s
			cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice); // copy input data to device

			// Pad the rest with 0s
            if (pow2 > n) {
				cudaMemset(dev_data + n, 0, (pow2 - n) * sizeof(int)); // set the rest to 0
            }
			checkCUDAError("cudaMemcpy H2D and padding for efficient scan"); // check for errors

            timer().startGpuTimer();
            // TODO

			const int blockSize = 128; // number of threads per block

            // Up-sweep (reduce) phase: build sum in place
            for (int step = 2; step <= pow2; step *= 2) {
                int threads = pow2 / step;        // number of add operations at this level
				int fullBlocks = (threads + blockSize - 1) / blockSize; // number of blocks needed
				kernUpSweep << <fullBlocks, blockSize >> > (pow2, step, dev_data); // launch kernel
				checkCUDAError("kernUpSweep kernel"); // check for errors
            }

            // Set the last element (total sum) to 0 for exclusive scan
			cudaMemset(dev_data + (pow2 - 1), 0, sizeof(int)); // set the last element to 0
			checkCUDAError("cudaMemset root (exclusive scan)"); // check for errors

            // Down-sweep phase: distribute the prefix sums
            for (int step = pow2; step >= 2; step /= 2) {

				// launch kernel
				int threads = pow2 / step; // number of add operations at this level
				int fullBlocks = (threads + blockSize - 1) / blockSize; // number of blocks needed
				kernDownSweep << <fullBlocks, blockSize >> > (pow2, step, dev_data); // launch kernel
				checkCUDAError("kernDownSweep kernel"); // check for errors
            }

            timer().endGpuTimer();

            // Read back the scanned result (first n elements)
			cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost); // copy result back to host
			checkCUDAError("cudaMemcpy D2H for efficient scan result"); // check for errors

			cudaFree(dev_data); // free device memory
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}
