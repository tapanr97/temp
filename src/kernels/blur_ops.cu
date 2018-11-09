#include <cuda_runtime.h>
#include <stdio.h>     
#include <math.h>

unsigned char *d_red, *d_green, *d_blue;
float *d_filter;
float *d_filter_vertical, *d_filter_horizontal;
uchar4 *d_inputImageRGBA__;
uchar4 *d_outputImageRGBA__;
float *h_filter__;

__global__ void gaussian_blur(const unsigned char* const inputChannel,
                   unsigned char* const outputChannel,
                   int numRows, int numCols, const float* const filter_vertical, const int filterWidth) {
  
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if ( col >= numCols || row >= numRows )
		return;

	float result = 0.f;
	
	//For every value in the filter around the pixel (c, r)
	for (int filter_r = -filterWidth/2; filter_r <= filterWidth/2; ++filter_r) {
			//Find the global image position for this filter position
			//clamp to boundary of the image
			int image_r = min(max(row + filter_r, 0), static_cast<int>(numRows - 1));

			float image_value = static_cast<float>(inputChannel[image_r * numCols + col]);
			float filter_value = filter_vertical[(filter_r + filterWidth/2)];

			result += image_value * filter_value;
		
	}
	outputChannel[row * numCols + col] = result;

}

__global__ void gaussian_blur2(const unsigned char* const inputChannel,
                   unsigned char* const outputChannel,
                   int numRows, int numCols,const float* const filter_horizontal, const int filterWidth) {
  
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if ( col >= numCols || row >= numRows )
		return;

	float result = 0.f;

	//For every value in the filter around the pixel (c, r)
	for (int filter_c = -filterWidth/2; filter_c <= filterWidth/2; ++filter_c) {
			//Find the global image position for this filter position
			//clamp to boundary of the image
			int image_c = min(max(col + filter_c, 0), static_cast<int>(numCols - 1));

			float image_value = static_cast<float>(inputChannel[image_c + row * numCols]);
			float filter_value = filter_horizontal[(filter_c + filterWidth/2)];

			result += image_value * filter_value;
		
	}
	outputChannel[row * numCols + col] = result;

}

__global__ void separateChannels(const uchar4* const inputImageRGBA,
                      int numRows,
                      int numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel) {

	int absolute_image_position_x = blockDim.x * blockIdx.x + threadIdx.x;
	int absolute_image_position_y = blockDim.y * blockIdx.y + threadIdx.y;

	if ( absolute_image_position_x >= numCols || absolute_image_position_y >= numRows )
		return;

	int thread_1D_pos = absolute_image_position_y * numCols + absolute_image_position_x;

	redChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].x;
	greenChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].y;
	blueChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].z;
}

__global__ void recombineChannels(const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const outputImageRGBA,
                       int numRows,
                       int numCols) {

	const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
	                                    blockIdx.y * blockDim.y + threadIdx.y);

	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
	return;

	unsigned char red   = redChannel[thread_1D_pos];
	unsigned char green = greenChannel[thread_1D_pos];
	unsigned char blue  = blueChannel[thread_1D_pos];

	//Alpha should be 255 for no transparency
	uchar4 outputPixel = make_uchar4(red, green, blue, 255);

	outputImageRGBA[thread_1D_pos] = outputPixel;
}




void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage, 
					const float* const h_filter_vertical, const float* const h_filter_horizontal, const size_t filterWidth) { 
	//allocate memory for the three different channels
  	//original
	cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage);
	cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage);
	cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage);

	cudaMalloc(&d_filter_vertical, sizeof(float)*filterWidth);
	cudaMalloc(&d_filter_horizontal, sizeof(float)*filterWidth);
	cudaMemcpy(d_filter_vertical,h_filter_vertical,sizeof(float)*filterWidth,cudaMemcpyHostToDevice);
	cudaMemcpy(d_filter_horizontal,h_filter_horizontal,sizeof(float)*filterWidth,cudaMemcpyHostToDevice);
}

void cleanup() {
	cudaFree(d_red);
	cudaFree(d_green);
	cudaFree(d_blue);
	cudaFree(d_filter);
}


void setFilter(float **h_filter_vertical, float **h_filter_horizontal, int *filterWidth, int blurKernelWidth, float blurKernelSigma) { 
	//Normally blurKernelWidth = 9 and blurKernelSigma = 2.0 
	*h_filter_vertical = new float[blurKernelWidth];
	*h_filter_horizontal = new float[blurKernelWidth];
	*filterWidth = blurKernelWidth;

	float filterSum = 0.f; //for normalization

	for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) {
	  		float filterValue = expf( -(float)(r * r ) / (2.f * blurKernelSigma * blurKernelSigma));
	  		(*h_filter_vertical)[(r + blurKernelWidth/2)] = filterValue;
	  		(*h_filter_horizontal)[(r + blurKernelWidth/2)] = filterValue;
	  		filterSum += filterValue;
	}

	float normalizationFactor = 1.f / filterSum;

	for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) {
	  	(*h_filter_vertical)[(r + blurKernelWidth/2)] *= normalizationFactor;
	  	(*h_filter_horizontal)[(r + blurKernelWidth/2)] *= normalizationFactor;	  	
	}

}

uchar4* blur_ops(uchar4* d_inputImageRGBA, size_t numRows, size_t numCols, int blurKernelWidth) { 
	
	float blurKernelSigma = blurKernelWidth/4.0f;
  	//Set filter array
	float* h_filter_vertical, *h_filter_horizontal;
	int filterWidth;
	setFilter(&h_filter_vertical, &h_filter_horizontal, &filterWidth, blurKernelWidth, blurKernelSigma);

	//Set reasonable block size (i.e., number of threads per block)
	const dim3 blockSize(16,16,1);
	//Calculate Grid SIze
	int a=numCols/blockSize.x, b=numRows/blockSize.y;	
	const dim3 gridSize(a+1,b+1,1);
	const size_t numPixels = numRows * numCols;

	uchar4 *d_outputImageRGBA;
	cudaMalloc((void **)&d_outputImageRGBA, sizeof(uchar4) * numPixels);
	cudaMemset(d_outputImageRGBA, 0, numPixels * sizeof(uchar4)); //make sure no memory is left laying around

	d_inputImageRGBA__  = d_inputImageRGBA;
	d_outputImageRGBA__ = d_outputImageRGBA;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//blurred
	unsigned char *d_redBlurred, *d_greenBlurred, *d_blueBlurred;
	unsigned char *d_redMid, *d_greenMid, *d_blueMid;
	cudaMalloc(&d_redBlurred,    sizeof(unsigned char) * numPixels);
	cudaMalloc(&d_greenBlurred,  sizeof(unsigned char) * numPixels);
	cudaMalloc(&d_blueBlurred,   sizeof(unsigned char) * numPixels);
	cudaMemset(d_redBlurred,   0, sizeof(unsigned char) * numPixels);
	cudaMemset(d_greenBlurred, 0, sizeof(unsigned char) * numPixels);
	cudaMemset(d_blueBlurred,  0, sizeof(unsigned char) * numPixels);

	cudaMalloc(&d_redMid,    sizeof(unsigned char) * numPixels);
	cudaMalloc(&d_greenMid,  sizeof(unsigned char) * numPixels);
	cudaMalloc(&d_blueMid,   sizeof(unsigned char) * numPixels);
	cudaMemset(d_redMid,   0, sizeof(unsigned char) * numPixels);
	cudaMemset(d_greenMid, 0, sizeof(unsigned char) * numPixels);
	cudaMemset(d_blueMid,  0, sizeof(unsigned char) * numPixels);

	allocateMemoryAndCopyToGPU(numRows, numCols, h_filter_vertical, h_filter_horizontal, filterWidth);

	cudaEventRecord(start, 0);

	//Launch a kernel for separating the RGBA image into different color channels
	separateChannels<<<gridSize, blockSize>>>(d_inputImageRGBA, numRows, numCols, d_red,d_green, d_blue);

	cudaDeviceSynchronize(); 

	//Call blur kernel here 3 times, once for each color channel.
	gaussian_blur<<<gridSize, blockSize>>>(d_red, d_redMid, numRows, numCols,  d_filter_vertical, filterWidth);
	gaussian_blur<<<gridSize, blockSize>>>(d_green, d_greenMid, numRows, numCols,  d_filter_vertical, filterWidth);
	gaussian_blur<<<gridSize, blockSize>>>(d_blue, d_blueMid, numRows, numCols,  d_filter_vertical, filterWidth);

	gaussian_blur2<<<gridSize, blockSize>>>(d_redMid, d_redBlurred, numRows, numCols, d_filter_horizontal, filterWidth);
	gaussian_blur2<<<gridSize, blockSize>>>(d_greenMid, d_greenBlurred, numRows, numCols,d_filter_horizontal, filterWidth);
	gaussian_blur2<<<gridSize, blockSize>>>(d_blueMid, d_blueBlurred, numRows, numCols,  d_filter_horizontal, filterWidth);
	cudaDeviceSynchronize(); 

	//Now we recombine the results.
	recombineChannels<<<gridSize, blockSize>>>(d_redBlurred,
	                                    	d_greenBlurred,
	                                        d_blueBlurred,
	                                        d_outputImageRGBA,
	                                        numRows,
	                                        numCols);
	
	cudaDeviceSynchronize(); 

	cudaEventRecord(stop, 0);
	
	cudaEventSynchronize(stop);
	
	float gpu_ms;
	cudaEventElapsedTime(&gpu_ms, start, stop);
	printf("GPU execution time for Gaussian Blur: %f\n", gpu_ms);

	//cleanup memory
	cleanup();
	cudaFree(d_redBlurred);
	cudaFree(d_greenBlurred);
	cudaFree(d_blueBlurred);
	cudaFree(d_redMid);
	cudaFree(d_greenMid);
	cudaFree(d_blueMid);


	cudaDeviceSynchronize(); 

	//Initialize memory on host for output uchar4*
	uchar4* h_out;
	h_out = (uchar4*)malloc(sizeof(uchar4) * numPixels);

	//copy output from device to host
	cudaMemcpy(h_out, d_outputImageRGBA, sizeof(uchar4) * numPixels, cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize(); 

	//cleanup memory on device
	cudaFree(d_inputImageRGBA__);
	cudaFree(d_outputImageRGBA__);
	delete[] h_filter__;

	//return h_out
	return h_out;
}
