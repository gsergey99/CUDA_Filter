/*********************************************
*   Project: Práctica de Computadores Avanzados 
*
*   Program name: cpu_kernel.cpp
*
*   Author: Sergio Jiménez
*
*   Date created: 18-12-2020
*
*   Porpuse: Gestión para la realización de varios filtros con la CPU
*
*   Revision History: Reflejado en el repositorio de GitHub
|*********************************************/

#include <thread>
#include <chrono>
#include <ctime>
#include <iostream>
#include <typeinfo>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/core.hpp>

#include <string.h>
#include <math.h>

#include "include/colours.h"


void sobelFilterCPU(cv::Mat src_image, cv::Mat dest_image, const int width, const int height){
    float result;

    for(int x = 1; x < src_image.rows-1; x++) {
        for(int y = 1; y < src_image.cols-1; y++) {

            float dy = (-1*src_image.data[(x-1)*height + (y-1)]) + (-2*src_image.data[x*height+(y-1)]) + (-1*src_image.data[(x+1)*height+(y-1)]) +
                (src_image.data[(x-1)*height + (y+1)]) + (2*src_image.data[x*height+(y+1)]) + (src_image.data[(x+1)*height+(y+1)]);
                
            float dx = (src_image.data[(x-1)*height + (y-1)]) + (2*src_image.data[(x-1)*height+y]) + (src_image.data[(x-1)*height+(y+1)]) +
            (-1*src_image.data[(x+1)*height + (y-1)]) + (-2*src_image.data[(x+1)*height+y]) + (-1*src_image.data[(x+1)*height+(y+1)]);
            
            
            result = sqrt((pow(dx,2))+ (pow(dy,2)));

            /*Noise suppression*/
            if (result > 255) result = 255;
            if (result < 0) result = 0;
            dest_image.at<uchar>(x,y) = result;
        }
    }
}

void sharpenFilterCPU(cv::Mat src_image, cv::Mat dest_image, const int width, const int height){
    float result;
    for(int x = 1; x < src_image.rows-1; x++) {
        for(int y = 1; y < src_image.cols-1; y++) {

        result = (0 * src_image.data[(x-1)*height + (y-1)]) + (-1 * src_image.data[(x-1)*height + y]) + (0 * src_image.data[(x-1)*height+(y+1)]) +
        (-1 * src_image.data[x*height+(y-1)]) + (5 * src_image.data[x*height+y]) +(-1 * src_image.data[x*height+(y+1)]) + 
        (0 * src_image.data[(x+1)*height +(y-1)]) + (-1 *src_image.data[(x+1)*height + y]) + (0 * src_image.data[(x+1)*height + (y+1)]);
        
        /*Noise supression*/
        if (result > 255) result = 255;
        if (result < 0) result = 0;
            
            dest_image.at<uchar>(x,y) = result;
        }
    }
}

int main(int argc, char *argv[]){

    int rows, cols;
    long memmory_used;
    cv::Mat src_image, dest_image_cpu;

    src_image = cv::imread(argv[1]);
    
    if (src_image.empty()){
        std::cout << FRED("[MANAGER] There is a problem reading the image: ")<< src_image << std::endl;
        exit(EXIT_FAILURE);
    }

    cv::cvtColor(src_image, src_image, cv::COLOR_RGB2GRAY);
    dest_image_cpu = cv::Mat::zeros(src_image.size(),src_image.type());

    cols = src_image.cols;
    rows = src_image.rows;

    std::cout << FCYN("[MANAGER] Using Image ") << argv[1] << FCYN(" | ROWS = ") <<  rows << FCYN(" COLS = ") << cols << std::endl;
    
    auto start_time = std::chrono::system_clock::now();
    if (strcmp(argv[2], "sobel") == 0)   sobelFilterCPU(src_image, dest_image_cpu, rows, cols);
    if (strcmp(argv[2], "sharpen") == 0)   sharpenFilterCPU(src_image, dest_image_cpu, rows, cols);

    std::chrono::duration<double> time_cpu = std::chrono::system_clock::now() - start_time;
    memmory_used = src_image.cols * src_image.cols * sizeof(unsigned char);

    cv::imshow("CPU Filter",dest_image_cpu);

    std::cout << FYEL("[MANAGER] Time CPU ") << time_cpu.count() * 1000 << FYEL(" milliseconds ") << std::endl;
    std::cout << FYEL("[MANAGER] Memory occupied by the picture is ") << memmory_used << FYEL(" Bytes") << std::endl;

    int k = cv::waitKey(0);

    return 0;
}


