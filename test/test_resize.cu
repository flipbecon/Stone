/**
 * Created by desmond <desmond.yao@buaa.edu.cn> on 2018-11-18
 */

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "./Utils/timer.h"
#include "./Utils/utils.h"
#include "../src/RESIZE/Resize_.cu"

using namespace std;
using namespace cv;

#define HEIGHT  4
#define WIDTH   4
#define CHANNEL 3

void processUsingOpenCvCpu(std::string nput_file, std::string output_file);
//void processUsingOpenCvGpu(std::string input_file, std::string output_file);
void processUsingCuda(std::string input_file, std::string output_file);

int main(int argc, char **argv) {

    bool useEpsCheck = false; // set true to enable perPixelError and globalError
    double perPixelError = 3;
    double globalError   = 10;

    const string input_file = argc >= 2 ? argv[1] : "../../data/image_small.jpg";
    const string output_file_OpenCvCpu = argc >= 3 ? argv[2] : "../../data/image_small_OpenCvCpu.jpg";
    const string output_file_OpenCvGpu = argc >= 4 ? argv[3] : "../../data/image_small_OpenCvGpu.jpg";
    const string output_file_Cuda = argc >= 5 ? argv[2] : "../../data/image_small_Cuda.jpg";

    for (int i=0; i<1; ++i) {
        processUsingOpenCvCpu(input_file, output_file_OpenCvCpu);
        //processUsingOpenCvGpu(input_file, output_file_OpenCvGpu);
        processUsingCuda(input_file, output_file_Cuda);
    }

    compareImages(output_file_OpenCvCpu, output_file_Cuda, useEpsCheck, perPixelError, globalError);

    return 0;
}

void processUsingOpenCvCpu(std::string input_file, std::string output_file) {
    //Read input image from the disk
    Mat input = imread(input_file, CV_LOAD_IMAGE_COLOR);
    if(input.empty())
    {
        std::cout<<"Image Not Found: "<< input_file << std::endl;
        return;
    }

    Mat output;

    GpuTimer timer;
    timer.Start();
    resize(input, output, Size(), .25, 0.25, CV_INTER_LINEAR); // downscale 4x on both x and y

    timer.Stop();
    printf("OpenCv Cpu code ran in: %f msecs.\n", timer.Elapsed());

    imwrite(output_file, output);
}
/*
void processUsingOpenCvGpu(std::string input_file, std::string output_file) {
    //Read input image from the disk
    Mat inputCpu = imread(input_file,CV_LOAD_IMAGE_COLOR);
    cuda::GpuMat input (inputCpu);
    if(input.empty())
    {
        std::cout<<"Image Not Found: "<< input_file << std::endl;
        return;
    }

    //Create output image
    cuda::GpuMat output;

    GpuTimer timer;
    timer.Start();

    cuda::resize(input, output, Size(), .25, 0.25, CV_INTER_AREA); // downscale 4x on both x and y

    timer.Stop();
    printf("OpenCv Gpu code ran in: %f msecs.\n", timer.Elapsed());

    Mat outputCpu;
    output.download(outputCpu);
    imwrite(output_file, outputCpu);

    input.release();
    output.release();
}
*/
void processUsingCuda(std::string input_file, std::string output_file) {
    //Read input image from the disk
    cv::Mat input = cv::imread(input_file, CV_LOAD_IMAGE_UNCHANGED);
    if(input.empty())
    {
        std::cout<<"Image Not Found: "<< input_file << std::endl;
        return;
    }

//    char data[HEIGHT * WIDTH * CHANNEL];
//
//    for (int i = 0; i < HEIGHT; ++i)
//    {
//        for(int j = 0; j < WIDTH; ++j)
//        {
//            data[i * WIDTH * CHANNEL + j*CHANNEL + 0] = i * WIDTH * CHANNEL + j*CHANNEL + 0;
//            data[i * WIDTH * CHANNEL + j*CHANNEL + 1] = i * WIDTH * CHANNEL + j*CHANNEL + 1;
//            data[i * WIDTH * CHANNEL + j*CHANNEL + 2] = i * WIDTH * CHANNEL + j*CHANNEL + 2;
//        }
//    }

//    cv::Mat point_mat(HEIGHT, WIDTH, CV_8UC3, data);

    //Create output image
    Size newSize( input.size().width / 4, input.size().height / 4 ); // downscale 4x on both x and y
    Mat output (newSize, input.type());

    downscaleCuda(input, output);

    imwrite(output_file, output);
}