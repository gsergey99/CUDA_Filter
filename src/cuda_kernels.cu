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


enum Options{IMAGE, FRAME}; //Option's enumeration

/*Global variables for the  sobel kernel's gradient*/
__constant__ int kernel_sobel_x[DIM_KERNEL][DIM_KERNEL]={{-1,0,1},{-2,0,2},{-1,0,1}};
__constant__ int kernel_sobel_y[DIM_KERNEL][DIM_KERNEL]={{-1,-2,-1},{0,0,0},{1,2,1}};

/*Global variable for the sharpen kernel*/
__constant__ int kernel_sharpen[DIM_KERNEL][DIM_KERNEL] = {{0,-1,0},{-1,5,-1},{0,-1,0}};

__global__ void sobelKernelCUDA(unsigned char* src_image, unsigned char* dest_image, int height, int width){
     
    float dx = 0;
    float dy = 0;
    float result;

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;


    for (int i = -1;i < DIM_KERNEL -1; i++){
      for(int j = -1;j < DIM_KERNEL - 1;j++ ){
          dx += (kernel_sobel_x[1+i][1+j] * src_image[(x+i)*width + (y+j)]);
          dy += (kernel_sobel_y[1+i][1+j] * src_image[(x+i)*width + (y+j)]);
      }
    }
   
    result = sqrt((pow(dx,2))+ (pow(dy,2)));

    /*Noise suppression*/
    if (result > MAX_VALUE_PIXEL) result = MAX_VALUE_PIXEL;
    if (result < MIN_VALUE_PIXEL) result = MIN_VALUE_PIXEL;

    dest_image[x*width+y] = result;

}

__global__ void sharpenKernelCUDA(unsigned char* src_image, unsigned char* dest_image, int height, int width){
   
    float result = 0;

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    for (int i = -1;i < DIM_KERNEL -1; i++){
      for(int j = -1;j < DIM_KERNEL - 1;j++ ){
          result += (kernel_sharpen[1+i][1+j] * src_image[(x+i)*width + (y+j)]);
      }
    }
    
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

void pictureFilter (char *filter, std::string image);
void frameFilter(char *filter, std::string video);
void cudaInit(char *filter, cv::Mat src_image, Options option);

int main(int argc,char * argv[]){
                
    if (argc > 4 && argc < 3){
        std::cout << FRED("[CUDA MANAGER] The number of arguments are incorrect, please insert <0:picture, 1 video, 2:live> <filter name: sobel, sharpen> <image or not image>  ") << std::endl;
        return EXIT_FAILURE;
    }

    if (strcmp(argv[2], "") == 0){
      std::cout << FRED("[CUDA MANAGER] The argument <filter name: sobel, sharpen> is not indicated") << std::endl;
      return EXIT_FAILURE;
    }
    
    std::string src_image;

    try{

      if (strcmp(argv[1], "0") == 0){ 

        src_image = argv[3];
        pictureFilter(argv[2], src_image);
      
      } else if (strcmp(argv[1], "1") == 0){

        src_image = argv[3];
        frameFilter(argv[2],src_image);

      } else if (strcmp(argv[1], "2") == 0){
        src_image = "";
        frameFilter(argv[2],src_image);
      }
      else{

        std::cout << FRED("[CUDA MANAGER] The argument <0:picture, 1 video, 2:live> is not indicated") << std::endl;
        return 1;
      }

    }catch(const std::exception &ex){
      
      std::cout << FRED("[MANAGER] There is some problem in the execution") << std::endl;    
      return EXIT_FAILURE;
    
    }
    
  return 0;
}


void pictureFilter (char* filter, std::string image){

    int rows, cols;
    cv::Mat src_image;
    Options option = IMAGE;

    src_image = cv::imread(image);
      
    if (src_image.empty()){
        std::cout << FRED("[CUDA MANAGER] There is a problem reading the image: ")<< src_image << std::endl;
        exit(EXIT_FAILURE);
    }

    cv::cvtColor(src_image, src_image, cv::COLOR_RGB2GRAY);

    cols = src_image.cols;
    rows = src_image.rows;

    std::cout << FCYN("[CUDA MANAGER] Using Image ") << image << FCYN(" | ROWS = ") <<  rows << FCYN(" COLS = ") << cols << std::endl;
    
    cudaInit(filter,src_image, option); //CUDA execution

    cv::resize(src_image, src_image, cv::Size(1360,700), 0.75, 0.75);
    
    cv::imshow("CUDA Filter Image",src_image);

    int k = cv::waitKey(0);

}

void frameFilter(char* filter, std::string video){

    int tick;
    cv::Mat src_image;
    cv::VideoCapture cap;
    std::time_t timeBegin = std::time(0);
    std::time_t timeEnd;
    Options option = FRAME;

    long frame_Counter = 0;

    tick = 0;

    if (video == ""){
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

      cudaInit(filter,src_image,option); //CUDA execution

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

void cudaInit(char *filter, cv::Mat src_image, Options option){

    int rows, cols, size;
    long memmory_used;
    unsigned char *d_image, *h_image;
    std::chrono::duration<double> time_gpu;
    auto start_time = std::chrono::system_clock::now();
    
    cols = src_image.cols;
    rows = src_image.rows;
    size = rows * cols;
    
    testCuErr(cudaMalloc((void**)&d_image, size));
    testCuErr(cudaMalloc((void**)&h_image, size));

    testCuErr(cudaMemcpy(d_image,src_image.data,size, cudaMemcpyHostToDevice));
    
    cudaMemset(h_image, 0, size);
    
    dim3 threadsPerBlock(N, N,1);
    dim3 numBlocks((int)ceil(rows/N), (int)ceil(cols/N),1);
    
    if (option == IMAGE) start_time = std::chrono::system_clock::now();
    if (strcmp(filter, "sobel") == 0)  sobelKernelCUDA <<<numBlocks, threadsPerBlock>>> (d_image, h_image, rows, cols);
    if (strcmp(filter, "sharpen") == 0) sharpenKernelCUDA <<<numBlocks, threadsPerBlock>>> (d_image, h_image, rows, cols);
    
    testCuErr(cudaGetLastError());
    if (option == IMAGE) time_gpu = std::chrono::system_clock::now() - start_time;

    cudaMemcpy(src_image.data, h_image, size, cudaMemcpyDeviceToHost);
    cudaFree(d_image); 
    cudaFree(h_image);

    memmory_used = src_image.cols * src_image.cols * sizeof(unsigned char);

    if (option == IMAGE) { 
      std::cout << FYEL("[CUDA MANAGER] Time GPU ") << time_gpu.count() * 1000000 << FYEL(" microseconds ") << std::endl;
      std::cout << FYEL("[CUDA MANAGER] Memory occupied by the picture is ") << memmory_used << FYEL(" Bytes") << std::endl;
    }
}