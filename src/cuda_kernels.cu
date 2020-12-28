/*********************************************
*   Project: Práctica de Computadores Avanzados 
*
*   Program name: cuda_kernels.cu
*
*   Author: Sergio Jiménez
*
*   Date created: 11-12-2020
*
*   Porpuse: Gestión para la realización de varios filtros de imagen en CUDA
*
*   Revision History: Reflejado en el repositorio de GitHub
|*********************************************/

#include <thread>
#include <chrono>
#include <ctime>
#include <iostream>
#include <typeinfo>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/core.hpp>
#include "opencv2/videoio.hpp"
#include <opencv2/video.hpp>

#include <string.h>
#include <math.h>

#include "../include/colours.h"

#define N 25.0 // Number of threads
#define DIM_KERNEL 3 // Dimension of kernel
#define MAX_VALUE_PIXEL 255 //Maximum value of a pixel
#define MIN_VALUE_PIXEL 0 // Minimum value of a pixel

/*Global variables for the  sobel kernel's gradient*/
__device__ const int kernel_sobel_x[DIM_KERNEL][DIM_KERNEL]={{-1,0,1},{-2,0,2},{-1,0,1}};
__device__ const int kernel_sobel_y[DIM_KERNEL][DIM_KERNEL]={{-1,-2,-1},{0,0,0},{1,2,1}};

/*Global variable for the sharpen kernel*/
__device__ const int kernel_sharpen[DIM_KERNEL][DIM_KERNEL] = {{0,-1,0},{-1,5,-1},{0,-1,0}};

__global__ void sobelKernelCUDA(unsigned char* src_image, unsigned char* dest_image, int height, int width){
     
    float dx,dy,result;

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    dx = (kernel_sobel_x[0][0] * src_image[(x-1)*width + (y-1)]) + (kernel_sobel_x[0][1] * src_image[(x-1)*width + y]) + (kernel_sobel_x[0][2] * src_image[(x-1)*width+(y+1)]) +
    (kernel_sobel_x[1][0] * src_image[x*width+(y-1)]) + (kernel_sobel_x[1][1] * src_image[x*width+y]) +(kernel_sobel_x[1][2] * src_image[x*width+(y+1)]) + 
    (kernel_sobel_x[2][0] * src_image[(x+1)*width +(y-1)]) + (kernel_sobel_x[2][1] *src_image[(x+1)*width + y]) + (kernel_sobel_x[2][2] * src_image[(x+1)*width + (y+1)]);
    
    dy = (kernel_sobel_y[0][0] * src_image[(x-1)*width + (y-1)]) + (kernel_sobel_y[0][1] * src_image[(x-1)*width + y]) + (kernel_sobel_y[0][2] * src_image[(x-1)*width+(y+1)]) +
    (kernel_sobel_y[1][0] * src_image[x*width+(y-1)]) + (kernel_sobel_y[1][1] * src_image[x*width+y]) +(kernel_sobel_y[1][2] * src_image[x*width+(y+1)]) + 
    (kernel_sobel_y[2][0] * src_image[(x+1)*width +(y-1)]) + (kernel_sobel_y[2][1] *src_image[(x+1)*width + y]) + (kernel_sobel_y[2][2] * src_image[(x+1)*width + (y+1)]);
    
    result = sqrt((pow(dx,2))+ (pow(dy,2)));

    /*Noise suppression*/
    if (result > MAX_VALUE_PIXEL) result = MAX_VALUE_PIXEL;
    if (result < MIN_VALUE_PIXEL) result = MIN_VALUE_PIXEL;

    dest_image[x*width+y] = result;

}

__global__ void sharpenKernelCUDA(unsigned char* src_image, unsigned char* dest_image, int height, int width){
   
    float result;

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    result = (kernel_sharpen[0][0] * src_image[(x-1)*width + (y-1)]) + (kernel_sharpen[0][1] * src_image[(x-1)*width + y]) + (kernel_sharpen[0][2] * src_image[(x-1)*width+(y+1)]) +
    (kernel_sharpen[1][0] * src_image[x*width+(y-1)]) + (kernel_sharpen[1][1] * src_image[x*width+y]) +(kernel_sharpen[1][2] * src_image[x*width+(y+1)]) + 
    (kernel_sharpen[2][0] * src_image[(x+1)*width +(y-1)]) + (kernel_sharpen[2][1] *src_image[(x+1)*width + y]) + (kernel_sharpen[2][2] * src_image[(x+1)*width + (y+1)]);
    
    /*Noise suppression*/
    if (result > MAX_VALUE_PIXEL) result = MAX_VALUE_PIXEL;
    if (result < MIN_VALUE_PIXEL) result = MIN_VALUE_PIXEL;

    dest_image[x*width+y] = result;
}


cudaError_t testCuErr(cudaError_t result){
  if (result != cudaSuccess) {
    printf(FCYN("[CUDA MANAGER] CUDA Runtime Error: %s\n"), cudaGetErrorString(result));
    assert(result == cudaSuccess);      
  }
  return result;
}

void pictureFilter (char* filter, char* image);
void frameFilter(char *filter, char* video);

int main(int argc,char * argv[]){
                
    if (argc > 4 && argc < 3){
        std::cout << FRED("[CUDA MANAGER] The number of arguments are incorrect, please insert <0:picture, 1 video, 2:live> <filter name: sobel, sharpen> <image or not image>  ") << std::endl;
        return 1;
    }

    if (strcmp(argv[2], "") == 0){
      std::cout << FRED("[CUDA MANAGER] The argument <filter name: sobel, sharpen> is not indicated") << std::endl;
      return 1;
    }
    
    if (strcmp(argv[1], "0") == 0){ 

      pictureFilter(argv[2], argv[3]);
    
    } else if (strcmp(argv[1], "1") == 0){

      frameFilter(argv[2],argv[3]);

    } else if (strcmp(argv[1], "2") == 0){

      frameFilter(argv[2],"");
    }
    else{

      std::cout << FRED("[CUDA MANAGER] The argument <0:picture, 1 video, 2:live> is not indicated") << std::endl;
      return 1;
    }
    
  return 0;
}


void pictureFilter (char* filter, char* image){

    int rows, cols;
    long memmory_used;
    cv::Mat src_image;

    src_image = cv::imread(image);
      
    if (src_image.empty()){
        std::cout << FRED("[CUDA MANAGER] There is a problem reading the image: ")<< src_image << std::endl;
        exit(EXIT_FAILURE);
    }

    cv::cvtColor(src_image, src_image, cv::COLOR_RGB2GRAY);

    cols = src_image.cols;
    rows = src_image.rows;

    std::cout << FCYN("[CUDA MANAGER] Using Image ") << image << FCYN(" | ROWS = ") <<  rows << FCYN(" COLS = ") << cols << std::endl;
    
    /*CUDA PART*/
    
    unsigned char *d_image, *h_image;
    int size = rows * cols;
    
    testCuErr(cudaMalloc((void**)&d_image, size));
    testCuErr(cudaMalloc((void**)&h_image, size));
    

    testCuErr(cudaMemcpy(d_image,src_image.data,size, cudaMemcpyHostToDevice));
    
    cudaMemset(h_image, 0, size);
    
    dim3 threadsPerBlock(N, N,1);
    dim3 numBlocks((int)ceil(rows/N), (int)ceil(cols/N),1);

    auto start_time = std::chrono::system_clock::now();
    if (strcmp(filter, "sobel") == 0)  sobelKernelCUDA <<<numBlocks, threadsPerBlock>>> (d_image, h_image, rows, cols);
    if (strcmp(filter, "sharpen") == 0) sharpenKernelCUDA <<<numBlocks, threadsPerBlock>>> (d_image, h_image, rows, cols);

    testCuErr(cudaGetLastError());
    std::chrono::duration<double> time_gpu = std::chrono::system_clock::now() - start_time;

    cudaMemcpy(src_image.data, h_image, size, cudaMemcpyDeviceToHost);
    cudaFree(d_image); 
    cudaFree(h_image);

    memmory_used = src_image.cols * src_image.cols * sizeof(unsigned char);

    cv::resize(src_image, src_image, cv::Size(1360,700), 0.75, 0.75);
    
    cv::imshow("CUDA Filter Image",src_image);

    std::cout << FYEL("[CUDA MANAGER] Time GPU ") << time_gpu.count() * 1000000 << FYEL(" microseconds ") << std::endl;
    std::cout << FYEL("[CUDA MANAGER] Memory occupied by the picture is ") << memmory_used << FYEL(" Bytes") << std::endl;

    int k = cv::waitKey(0);

}

void frameFilter(char * filter, char* video){

    int rows, cols, tick;
    cv::Mat src_image, dst_image;
    cv::VideoCapture cap;
    std::time_t timeBegin = std::time(0);
    std::time_t timeEnd;

    long frame_Counter = 0;

    tick = 0;

    if (strcmp(video, "") == 0){
      cap.open(0);
    } else{
      cap.open(video);
    }

    if(!cap.isOpened()){
      std::cout << FRED("[CUDA MANAGER] There is a problem catching the frame")<< src_image << std::endl;
      return exit(EXIT_FAILURE);
    }

    for(;;){

      cap >> src_image;

      if (src_image.empty()){
        break;
      }

      cv::cvtColor(src_image, src_image, cv::COLOR_RGB2GRAY);
      dst_image = cv::Mat::zeros(src_image.size(), src_image.type());
      
      cols = src_image.cols;
      rows = src_image.rows;
 
      unsigned char *d_image, *h_image;
      int size = rows * cols;
      
      testCuErr(cudaMalloc((void**)&d_image, size));
      testCuErr(cudaMalloc((void**)&h_image, size));

      testCuErr(cudaMemcpy(d_image,src_image.data,size, cudaMemcpyHostToDevice));
      cudaMemset(h_image, 0, size);
      
      dim3 threadsPerBlock(N, N,1);
      dim3 numBlocks((int)ceil(rows/N), (int)ceil(cols/N),1);

      if (strcmp(filter, "sobel") == 0)  sobelKernelCUDA <<<numBlocks, threadsPerBlock>>> (d_image, h_image, rows, cols);
      if (strcmp(filter, "sharpen") == 0) sharpenKernelCUDA <<<numBlocks, threadsPerBlock>>> (d_image, h_image, rows, cols);

      testCuErr(cudaGetLastError());

      cudaMemcpy(src_image.data, h_image, size, cudaMemcpyDeviceToHost);
      cudaFree(d_image); 
      cudaFree(h_image);

      cv::resize(src_image, src_image, cv::Size(1360,700), 0.75, 0.75);
      cv::imshow("CUDA Filter Frame",src_image);
      frame_Counter++;

      timeEnd = std::time(0) - timeBegin;

      if (timeEnd - tick >= 1)
      {
          tick++;
          std::cout << FMAG("[CUDA MANAGER] Frames per second: ") << frame_Counter << std::endl;
          frame_Counter = 0;
      }
      int k = cv::waitKey(33);
    }
}

