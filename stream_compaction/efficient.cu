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

			// Edge case: if n <= 0, return 0
            if (n <= 0) {
                timer().endGpuTimer();
                return 0;
            }

            // Device memory allocation
			int* dev_idata = nullptr; // device input array
			int* dev_bools = nullptr; // device boolean array
			int* dev_indices = nullptr; // device indices array
			int* dev_odata = nullptr; // device output array

			cudaMalloc(&dev_idata, n * sizeof(int)); // allocate device memory for input
			cudaMalloc(&dev_bools, n * sizeof(int)); // allocate device memory for boolean array
			cudaMalloc(&dev_odata, n * sizeof(int)); // allocate device memory for output
            checkCUDAError("cudaMalloc failed for compaction arrays");

            // We will allocate dev_indices with padding for scan
			int pow2 = 1 << ilog2ceil(n); // next power of 2
			cudaMalloc(&dev_indices, pow2 * sizeof(int)); // allocate device memory for indices with padding
			checkCUDAError("cudaMalloc failed for indices array"); // check for errors

            // Copy input to device
			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice); // copy input data to device
			checkCUDAError("cudaMemcpy H2D for compaction input"); // check for errors

            // Map input to booleans (1 = keep, 0 = discard)
			int blockSize = 128; // number of threads per block
			int fullBlocks = (n + blockSize - 1) / blockSize; // number of blocks needed
			StreamCompaction::Common::kernMapToBoolean <<<fullBlocks, blockSize >>> (n, dev_bools, dev_idata); // launch kernel
			checkCUDAError("kernMapToBoolean kernel"); // check for errors

            // Scan on dev_bools -> dev_indices (inclusive of padding)
            // Copy bools to indices array (and pad remaining space with 0)
			cudaMemcpy(dev_indices, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice); // copy bools to indices

            if (pow2 > n) {
				cudaMemset(dev_indices + n, 0, (pow2 - n) * sizeof(int)); // pad remaining space with 0
            }
            checkCUDAError("copy + pad boolean array for scan");

            // Up-sweep phase on indices array
            for (int step = 2; step <= pow2; step *= 2) {

				int threads = pow2 / step; // number of add operations at this level
				fullBlocks = (threads + blockSize - 1) / blockSize; // number of blocks needed
				kernUpSweep <<<fullBlocks, blockSize >>> (pow2, step, dev_indices); // launch kernel
                checkCUDAError("kernUpSweep (compaction) kernel");
            }

            // Set last element to 0 (prepare for exclusive scan)
			cudaMemset(dev_indices + (pow2 - 1), 0, sizeof(int)); // set last element to 0
            checkCUDAError("cudaMemset root for compaction scan");

            // Down-sweep phase
            for (int step = pow2; step >= 2; step /= 2) {

				int threads = pow2 / step; // number of add operations at this level
				fullBlocks = (threads + blockSize - 1) / blockSize; // number of blocks needed
				kernDownSweep <<<fullBlocks, blockSize >>> (pow2, step, dev_indices); // launch kernel
                checkCUDAError("kernDownSweep (compaction) kernel");
            }

            // Scatter non-zero elements to output array using computed indices
			fullBlocks = (n + blockSize - 1) / blockSize; // number of blocks needed
            StreamCompaction::Common::kernScatter <<<fullBlocks, blockSize >>> (
				n, dev_odata, dev_idata, dev_bools, dev_indices); // launch kernel
            checkCUDAError("kernScatter kernel");


            timer().endGpuTimer();
            
            // Copy compacted data back to host
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy D2H for compaction output");

            // Compute and return count of non-zero elements
            int count = 0;
            int lastBool, lastIndex;
            cudaMemcpy(&lastBool, dev_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastIndex, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy D2H for compaction count");
            if (n > 0) {
                count = lastIndex + lastBool;
            }

            cudaFree(dev_idata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);
            cudaFree(dev_odata);
            return count;
        }
    }
}
