#include "../cMat.h"
#include <stdio.h>
#include <cmath>
#include <iostream>


int main(int argc, char** argv ){
    //printOclPlatformInfo();

    // Generate random matrix
    cv::Mat rMat(cv::Size(200,200), 5);
    double low = -500.0;
    double high = +500.0;
    cv::randu(rMat, cv::Scalar(low), cv::Scalar(high));

    cMat mat1(rMat);
    mat1 += 1.0;
    cmshow(mat1,"title");
}
