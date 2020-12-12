/*********************************************
*   Project: Práctica de Computadores Avanzados 
*
*   Program name: sobel_filter.cu
*
*   Author: Sergio Jiménez
*
*   Date created: 11-12-2020
*
*   Porpuse: Gestión para la realización de un filtro Sobel
*
*   Revision History: Reflejado en el repositorio de GitHub
|*********************************************/

#include <thread>
#include <chrono>
#include <time.h>
#include <iostream>
#include <math.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/core.hpp>

#define N 100000 // Number of threads
#define tb 512  // Block size
#define DIM_KERNEL 3 // Dimention of the kernel

__global__ void sobelKernel(unsigned char * src_image, int width, int height){
    const int kernel_x[DIM_KERNEL][DIM_KERNEL]={{-1,0,1},{-2,0,2},{-1,0,1}};
    const int kernel_y[DIM_KERNEL][DIM_KERNEL]={{-1,-2,-1},{0,0,0},{1,2,1}};
    int x,y,dx,dy,i,j, result;
    dx = 0;
    dy = 0;
    
    //Pensar en un código de manera secuencial
    result = sqrt((pow(dx,2))+ (pow(dy,2)));


    }




cudaError_t testCuErr(cudaError_t result){
  if (result != cudaSuccess) {
    printf("CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);      // si no se cumple, se aborta el programa
  }
  return result;
}


int main(int argc,char * argv[]){
    
    int rows, cols;
    
    if (argc!=2){
        std::cout << "[MANAGER] The number of arguments are incorrect, please insert <image>" << std::endl;
        return 1;
    }

    cv::Mat src_image = cv::imread(argv[1]);
    if (src_image.empty()){
        std::cout << "[MANAGER] There is a problem reading the image "<< src_image << std::endl;
        return 1;
    }

    cv::cvtColor(src_image, src_image, cv::COLOR_RGB2GRAY);
    
    cols = src_image.cols;
    rows = src_image.rows;

    std::cout << "[MANAGER] Using Image " << argv[1] << " | ROWS = " <<  rows << " COLS = " << cols << std::endl;
    
    unsigned char *d_image, *h_image;
    int size = rows * cols * sizeof(unsigned char);
    h_image = (unsigned char*) malloc(size);

    testCuErr(cudaMalloc((void**)&d_image,size));
    testCuErr(cudaMemcpy(d_image,src_image.data,rows*cols*sizeof(unsigned char), cudaMemcpyHostToDevice));

    int dg = (N+tb-1)/tb; if (dg>65535) dg=65535;



    //cv::imshow("Sobel Filter", src_image);





    //int k = cv::waitKey(0); // Wait for a keystroke in the window

    return 0;

}