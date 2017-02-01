#include "../cMat.h"
#include <stdio.h>
#include <cmath>
#include <iostream>

using namespace cvc;

int main(int argc, char** argv ){
    //printOclPlatformInfo();

    // Generate random matrix
    cv::UMat testMat(cv::Size(5,5), 5);
    cv::UMat testMat2(cv::Size(5,5), 5);
    double low = 0.0;
    double high = +1.0;
    cv::randu(testMat, cv::Scalar(low), cv::Scalar(high));
    cv::randu(testMat2, cv::Scalar(low), cv::Scalar(high));

    //testMat2 = cv::UMat::zeros(cv::Size(5,5), 5);

    cMat mat_complex(testMat,testMat2);
    mat_complex.real = testMat;
    mat_complex.imag = testMat2;
    std::cout << "Complex initialization - size: " << mat_complex.size() << std::endl;

    // Print first indicies of array
    std::cout << "real: " << mat_complex.real.getMat(cv::ACCESS_READ).at<float>(0,0) <<std::endl;
    std::cout << "imag: " << mat_complex.imag.getMat(cv::ACCESS_READ).at<float>(0,0) <<std::endl;


    cmshow(mat_complex,"Complex Mat");


}
