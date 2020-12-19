/*********************************************
*   Project: Práctica de Computadores Avanzados 
*
*   Program name: cuda_kernels_live.cu
*
*   Author: Sergio Jiménez
*
*   Date created: 16-12-2020
*
*   Porpuse: Gestión para la realización de varios filtros en vivo
*
*   Revision History: Reflejado en el repositorio de GitHub
|*********************************************/

#include <thread>
#include <chrono>
#include <string.h>
#include <ctime>
#include <iostream>
#include <math.h>
#include <typeinfo>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/core.hpp>
#include "../include/colours.h"

#define N 25.0 // Number of threads
#define DIM_KERNEL 3 // Dimension of kernel

/*Global variables for the  sobel kernel's gradient*/
__device__ const int kernel_sobel_x[DIM_KERNEL][DIM_KERNEL]={{-1,0,1},{-2,0,2},{-1,0,1}};
__device__ const int kernel_sobel_y[DIM_KERNEL][DIM_KERNEL]={{-1,-2,-1},{0,0,0},{1,2,1}};

__device__ const int kernel_sharpen[DIM_KERNEL][DIM_KERNEL] = {{0,-1,0},{-1,5,-1},{0,-1,0}};

__global__ void sobelKernelCUDA(unsigned char* src_image, unsigned char* dest_image, int width, int height){
     
    float dx,dy,result;

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    dx = (kernel_sobel_x[0][0] * src_image[(x-1)*height + (y-1)]) + (kernel_sobel_x[0][1] * src_image[(x-1)*height + y]) + (kernel_sobel_x[0][2] * src_image[(x-1)*height+(y+1)]) +
    (kernel_sobel_x[1][0] * src_image[x*height+(y-1)]) + (kernel_sobel_x[1][1] * src_image[x*height+y]) +(kernel_sobel_x[1][2] * src_image[x*height+(y+1)]) + 
    (kernel_sobel_x[2][0] * src_image[(x+1)*height +(y-1)]) + (kernel_sobel_x[2][1] *src_image[(x+1)*height + y]) + (kernel_sobel_x[2][2] * src_image[(x+1)*height + (y+1)]);
    
    dy = (kernel_sobel_y[0][0] * src_image[(x-1)*height + (y-1)]) + (kernel_sobel_y[0][1] * src_image[(x-1)*height + y]) + (kernel_sobel_y[0][2] * src_image[(x-1)*height+(y+1)]) +
    (kernel_sobel_y[1][0] * src_image[x*height+(y-1)]) + (kernel_sobel_y[1][1] * src_image[x*height+y]) +(kernel_sobel_y[1][2] * src_image[x*height+(y+1)]) + 
    (kernel_sobel_y[2][0] * src_image[(x+1)*height +(y-1)]) + (kernel_sobel_y[2][1] *src_image[(x+1)*height + y]) + (kernel_sobel_y[2][2] * src_image[(x+1)*height + (y+1)]);
    
    result = sqrt((pow(dx,2))+ (pow(dy,2)));

    /*Noise suppression*/
    if (result > 255) result = 255;
    if (result < 0) result = 0;

    dest_image[x*height+y] = (int)result;

}

__global__ void sharpenKernelCUDA(unsigned char* src_image, unsigned char* dest_image, int width, int height){
   
  float result;

  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  
  result = (kernel_sharpen[0][0] * src_image[(x-1)*height + (y-1)]) + (kernel_sharpen[0][1] * src_image[(x-1)*height + y]) + (kernel_sharpen[0][2] * src_image[(x-1)*height+(y+1)]) +
  (kernel_sharpen[1][0] * src_image[x*height+(y-1)]) + (kernel_sharpen[1][1] * src_image[x*height+y]) +(kernel_sharpen[1][2] * src_image[x*height+(y+1)]) + 
  (kernel_sharpen[2][0] * src_image[(x+1)*height +(y-1)]) + (kernel_sharpen[2][1] *src_image[(x+1)*height + y]) + (kernel_sharpen[2][2] * src_image[(x+1)*height + (y+1)]);
  
  /*Noise suppression*/
  if (result > 255) result = 255;
  if (result < 0) result = 0;

  dest_image[x*height+y] = (int)result;

}

cudaError_t testCuErr(cudaError_t result){
  if (result != cudaSuccess) {
    printf(FCYN("[CUDA MANAGER] CUDA Runtime Error: %s\n"), cudaGetErrorString(result));
    assert(result == cudaSuccess);      
  }
  return result;
}

int main(int argc,char * argv[]){
    
    int rows, cols, tick;
    cv::Mat src_image, dst_image;
    cv::VideoCapture cap;
    std::time_t timeBegin = std::time(0);
    std::time_t timeEnd;
    long frame_Counter = 0;

    tick = 0;

    if (argc!=2){
        std::cout << FRED("[MANAGER] The number of arguments are incorrect, please insert <image> <filter name: sobel, sharpen>") << std::endl;
        return 1;
    }

    if (strcmp(argv[1], "") == 0){
      std::cout << FRED("[MANAGER] The <filter name: sobel, sharpen> is not indicated") << std::endl;
      return 1;
    }
    if(!cap.open(0))
        return 0;

    for(;;){
      
      cap.read(src_image);

      if (src_image.empty()){
          std::cout << FRED("[MANAGER] There is a problem with the web cam")<< src_image << std::endl;
          return 1;
      }

      cv::cvtColor(src_image, src_image, cv::COLOR_RGB2GRAY);
      dst_image = cv::Mat::zeros(src_image.size(), src_image.type());
      
      cols = src_image.cols;
      rows = src_image.rows;

      /*CUDA PART*/
      
      unsigned char *d_image, *h_image;
      int size = rows * cols;
      
      testCuErr(cudaMalloc((void**)&d_image, size));
      testCuErr(cudaMalloc((void**)&h_image, size));

      testCuErr(cudaMemcpy(d_image,src_image.data,size, cudaMemcpyHostToDevice));
      cudaMemset(h_image, 0, size);
      
      dim3 threadsPerBlock(N, N, 1);
      dim3 numBlocks((int)ceil(rows/N), (int)ceil(cols/N), 1);

      if (strcmp(argv[1], "sobel") == 0)  sobelKernelCUDA <<<numBlocks, threadsPerBlock>>> (d_image, h_image, rows, cols);
      if (strcmp(argv[1], "sharpen") == 0) sharpenKernelCUDA <<<numBlocks, threadsPerBlock>>> (d_image, h_image, rows, cols);


        testCuErr(cudaGetLastError());

      cudaMemcpy(src_image.data, h_image, size, cudaMemcpyDeviceToHost);
      cudaFree(d_image); 
      cudaFree(h_image);

      
      cv::imshow("CUDA Filter LIVE",src_image);
      frame_Counter++;

      timeEnd = std::time(0) - timeBegin;

      if (timeEnd - tick >= 1)
      {
          tick++;
          std::cout << "Frames per second: " << frame_Counter << std::endl;
          frame_Counter = 0;
      }


      int k = cv::waitKey(10); // Wait for a keystroke in the windows
      if(k ==27) break;
    }
    return 0;
  }
