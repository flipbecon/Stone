#ifndef RESIZE_CUH_
#define BILATERAL_FILTER_CUH_

#include<iostream>
#include<cstdio>

using std::cout;
using std::endl;

static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
	if(err!=cudaSuccess)
	{
		fprintf(stderr,"%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n",msg,file_name,line_number,cudaGetErrorString(err));
		std::cin.get();
		exit(EXIT_FAILURE);
	}
}

#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)

__global__ void resizeCudaKernel( unsigned char* input,
								  unsigned char* output,
								  const int outputWidth,
								  const int outputHeight,
								  const int inputWidth,
								  const int inputHeight,
								  const float pixelGroupSizeX,
								  const float pixelGroupSizeY,
								  const int inputChannels)
{
	//2D Index of current thread
	const int dx = blockIdx.x * blockDim.x + threadIdx.x;
	const int dy = blockIdx.y * blockDim.y + threadIdx.y;

	const int pitchInput = inputWidth * inputChannels;
	const int pitchOutput = outputWidth * inputChannels;

	if((dx < outputWidth) && (dy < outputHeight))
	{
		if(inputChannels==1) { // grayscale image
		} else if(inputChannels==3) { // RGB image

            float scale_x = (float) inputWidth / (float) outputWidth;
            float scale_y = (float) inputHeight / (float) outputHeight;

            float fx = ((float) dx + 0.5) * scale_x - 0.5;
            float sx = floor(fx);
            fx = fx - sx;
            int isx1 = (int) sx;
            if (isx1 < 0) {
                fx = 0.0;
                isx1 = 0;
            }
            if (isx1 > (inputWidth - 1)) {
                fx = 0;
                isx1 = inputWidth - 1;
            }

            float fy = ((float) dy + 0.5) * scale_y - 0.5;
            float sy = floor(fy);
            fy = fy - sy;
            int isy1 = (int) sy;
            if (isy1 < 0) {
                fy = 0.0;
                isy1 = 0;
            }
            if (sy > (inputHeight - 1)) {
                fy = 0.0;
                isy1 = inputHeight - 1;
            }

            int isx2 = min(isx1 + 1, inputWidth - 1);
            int isy2 = min(isy1 + 1, inputHeight - 1);

            float3 d0;

            float3 s11 = make_float3(input[(isy1 * inputWidth + isx1) * inputChannels + 0] , input[(isy1 * inputWidth + isx1) * inputChannels + 1] , input[(isy1 * inputWidth + isx1) * inputChannels + 2]);
            float3 s12 = make_float3(input[(isy1 * inputWidth + isx2) * inputChannels + 0] , input[(isy1 * inputWidth + isx2) * inputChannels + 1] , input[(isy1 * inputWidth + isx2) * inputChannels + 2]);
            float3 s21 = make_float3(input[(isy2 * inputWidth + isx1) * inputChannels + 0] , input[(isy2 * inputWidth + isx1) * inputChannels + 1] , input[(isy2 * inputWidth + isx1) * inputChannels + 2]);
            float3 s22 = make_float3(input[(isy2 * inputWidth + isx2) * inputChannels + 0] , input[(isy2 * inputWidth + isx2) * inputChannels + 1] , input[(isy2 * inputWidth + isx2) * inputChannels + 2]);

            float h_rst00, h_rst01;
            // B
            h_rst00 = s11.x * (1 - fx) + s12.x * fx;
            h_rst01 = s21.x * (1 - fx) + s22.x * fx;
            // d0.x = h_rst00 * (1 - fy) + h_rst01 * fy;
            d0.x = min(255, __float2uint_rn( h_rst00 * (1 - fy) + h_rst01 * fy ));
            // G
            h_rst00 = s11.y * (1 - fx) + s12.y * fx;
            h_rst01 = s21.y * (1 - fx) + s22.y * fx;
            // d0.y = h_rst00 * (1 - fy) + h_rst01 * fy;
            d0.y = min(255, __float2uint_rn( h_rst00 * (1 - fy) + h_rst01 * fy ));
            // R
            h_rst00 = s11.z * (1 - fx) + s12.z * fx;
            h_rst01 = s21.z * (1 - fx) + s22.z * fx;
            // d0.z = h_rst00 * (1 - fy) + h_rst01 * fy;
            d0.z = min(255, __float2uint_rn( h_rst00 * (1 - fy) + h_rst01 * fy ));

            output[(dy*outputWidth + dx) * 3 + 0 ] = static_cast<unsigned char>(d0.x); // R
            output[(dy*outputWidth + dx) * 3 + 1 ] = static_cast<unsigned char>(d0.y); // G
            output[(dy*outputWidth + dx) * 3 + 2 ] = static_cast<unsigned char>(d0.z); // B

		} else {

		}
	}
}

void downscaleCuda(const cv::Mat& input, cv::Mat& output)
{
	//Calculate total number of bytes of input and output image
	const int inputBytes = input.step * input.rows;
	const int outputBytes = output.step * output.rows;

	unsigned char *d_input, *d_output;

	//Allocate device memory
	SAFE_CALL(cudaMalloc<unsigned char>(&d_input,inputBytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<unsigned char>(&d_output,outputBytes), "CUDA Malloc Failed");

	//Copy data from OpenCV input image to device memory
	SAFE_CALL(cudaMemcpy(d_input,input.ptr(),inputBytes,cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");

	GpuTimer timer;
	timer.Start();

	//Specify a reasonable block size
	const dim3 block(16,16);

	//Calculate grid size to cover the whole image
	const dim3 grid((output.cols + block.x - 1) / block.x, (output.rows + block.y - 1) / block.y);

	// Calculate how many pixels in the input image will be merged into one pixel in the output image
	const float pixelGroupSizeY = float(input.rows) / float(output.rows);
	const float pixelGroupSizeX = float(input.cols) / float(output.cols);

	//Launch the size conversion kernel
	resizeCudaKernel<<<grid,block>>>(d_input, d_output, output.cols, output.rows, input.cols, input.rows, pixelGroupSizeX, pixelGroupSizeY, input.channels());

	timer.Stop();
	printf("Own Cuda code ran in: %f msecs.\n", timer.Elapsed());

	//Synchronize to check for any kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");

	//Copy back data from destination device meory to OpenCV output image
	SAFE_CALL(cudaMemcpy(output.ptr(),d_output,outputBytes,cudaMemcpyDeviceToHost),"CUDA Memcpy Host To Device Failed");

	//Free the device memory
	SAFE_CALL(cudaFree(d_input),"CUDA Free Failed");
	SAFE_CALL(cudaFree(d_output),"CUDA Free Failed");
	//SAFE_CALL(cudaDeviceReset(),"CUDA Device Reset Failed");
}

#endif /* RESIZE_CUH_ */
