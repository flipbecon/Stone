/*
 * resize.cuh
 *
 *  Created on: Dec 4, 2015
 *      Author: claudiu
 */

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
                                  const float scale_x,
                                  const float scale_y,
                                  const int inputChannels)
{
    //2D Index of current thread
    const int dx = blockIdx.x * blockDim.x + threadIdx.x;
    const int dy = blockIdx.y * blockDim.y + threadIdx.y;

    const int pitchInput = inputWidth * inputChannels;

    //Only valid threads perform memory I/O
    if((dx<outputWidth) && (dy<outputHeight))
    {
        // Starting location of current pixel in output

        if(inputChannels==1) { // grayscale image
        } else if(inputChannels==3) { // RGB image

            float fx = ( (float)dx + 0.5f)*scale_x - 0.5f;
            float sx = floor(fx);
            fx = fx - sx;
            int isx1 = (int)sx;
            if ( isx1 < 0 ){
                fx = 0.f;
                isx1 = 0;
            }
            if ( isx1 > (inputWidth-1) ){
                fx = 0;
                isx1 = inputWidth - 1;
            }

            float fy = ( (float)dy + 0.5f)*scale_y - 0.5f;
            float sy = floor(fy);
            fy = fy - sy;
            int isy1 = (int)sy;
            if (isy1 < 0){
                fy = 0.f;
                isy1 = 0;
            }
            if (sy > (inputHeight-1) ){
                fy = 0.f;
                isy1 = inputHeight-1;
            }

            int isx2 = min(isx1+1, inputWidth-1);
            int isy2 = min(isy1+1, inputHeight-1);

            unsigned char *a = input + isy1 * pitchInput + isx1 * inputChannels;
            unsigned char *b = input + isy1 * pitchInput + isx2 * inputChannels;
            unsigned char *c = input + isy2 * pitchInput + isx1 * inputChannels;
            unsigned char *d = input + isy2 * pitchInput + isx2 * inputChannels;

            int outputindex = dy * outputWidth * inputChannels + (dx * inputChannels);

            float h_rst00, h_rst01;
            // B
            h_rst00 = a[0] * (1 - fx) + b[0] * fx;
            h_rst01 = c[0] * (1 - fx) + d[0] * fx;
            output[outputindex] = min(255, __float2uint_rn( h_rst00 * (1 - fy) + h_rst01 * fy ));
            // G
            h_rst00 = a[1] * (1 - fx) + b[1] * fx;
            h_rst01 = c[1] * (1 - fx) + d[1] * fx;
            output[outputindex + 1] = min(255, __float2uint_rn( h_rst00 * (1 - fy) + h_rst01 * fy ));
            // R
            h_rst00 = a[2] * (1 - fx) + b[2] * fx;
            h_rst01 = c[2] * (1 - fx) + d[2] * fx;
            output[outputindex + 2] = min(255, __float2uint_rn( h_rst00 * (1 - fy) + h_rst01 * fy ));

        } else { // arbitrary number of channels
            // Compute the pixel group area in the input image
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
    SAFE_CALL(cudaMalloc<unsigned char>(&d_input,inputBytes),"CUDA Malloc Failed");
    SAFE_CALL(cudaMalloc<unsigned char>(&d_output,outputBytes),"CUDA Malloc Failed");

    //Copy data from OpenCV input image to device memory
    SAFE_CALL(cudaMemcpy(d_input,input.ptr(),inputBytes,cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");

    GpuTimer timer;
    timer.Start();

    //Specify a reasonable block size
    const dim3 block(16,16);

    //Calculate grid size to cover the whole image
    const dim3 grid((output.cols + block.x - 1)/block.x, (output.rows + block.y - 1)/block.y);

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
