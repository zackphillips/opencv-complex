#include "../cMat.h"
#include <stdio.h>
#include <cmath>
#include <iostream>

using namespace cvc;

cMat* getMat(int m, int n, double low, double high) {
    cv::UMat real(cv::Size(m, n), 5);
    cv::UMat im(cv::Size(m, n), 5);
    cv::randu(real, cv::Scalar(low), cv::Scalar(high));
    cv::randu(im, cv::Scalar(low), cv::Scalar(high));
    return new cMat(real, im);
}

cMat* getMat(int m, int n) {
    return getMat(m, n, 0.0, +1.0);
}

cMat* getMat(int n) {
    return getMat(n, n);
}

//test toString. 4x2 matrix.
void testToString() {
    std::cout << "Testing toString." << std::endl;
    double low = 0.0;
    double high = 1.0;
    cv::UMat toMatR(cv::Size(2, 4), 5);
    cv::UMat toMatI(cv::Size(2, 4), 5);
    cv::randu(toMatR, cv::Scalar(low), cv::Scalar(high));
    cv::randu(toMatI, cv::Scalar(low), cv::Scalar(high));
    cMat toMat(toMatR, toMatI);
    toMat.real = toMatR;
    toMat.imag = toMatI;
    std::cout << "toString test: " << '\n' << toMat.toString() << std::endl;
}

//test get
void testGet() {
    std::cout << "Testing Get." << std::endl;
    int size = 5;
    cMat* A = getMat(size);
    std::cout << "A is: " << '\n' << A->toString() << std::endl;
    for (int r = 0; r < size; r++) {
        for (int c = 0; c < size; c++) {
            std::complex<double>* z = A->get(r, c);
            std::cout << "A(" << r << ", " << c << ") is: " << z->real() <<
                " + i*" << z->imag() << std::endl;
        }
    }
}

//test set
void testSet() {
    std::cout << "Testing Set." << std::endl;
    int size = 5;
    cMat* A = getMat(size);
    std::cout << "A is: " << '\n' << A->toString() << std::endl;
    for (int r = 0; r < size; r++) {
        for (int c = 0; c < size; c++) {
            std::complex<double> z (r, c);
            A->set(r, c, z);
        }
    }
    std::cout << "A is: " << '\n' << A->toString() << std::endl;
}

// add test

// subtraction test

// multiplication test

// division test

int main(int argc, char** argv ){

    testToString();
    testGet();
    testSet();

    /*
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
    std::cout << "Complex initialization - size: " << mat_complex.size() << std::endl;

    // Print first indicies of array
//    std::cout << "real: " << mat_complex.real.getMat(cv::ACCESS_READ).at<float>(0,0) <<std::endl;
//    std::cout << "imag: " << mat_complex.imag.getMat(cv::ACCESS_READ).at<float>(0,0) <<std::endl;

    //Generate other random matrix
    cv::UMat otherMat(cv::Size(5, 5), 5);
    cv::UMat otherMat2(cv::Size(5, 5), 5);
    cv::randu(otherMat, cv::Scalar(low), cv::Scalar(high));
    cv::randu(otherMat2, cv::Scalar(low), cv::Scalar(high));
    cMat matComplex2(otherMat, otherMat2);
    matComplex2.real = otherMat;
    matComplex2.imag = otherMat2;

//    std::cout << "First matrix at 0, 0 has real part " << mat_complex.real.getMat(cv::ACCESS_READ).at<float>(0, 0)
//        << " and imaginary part " << mat_complex.imag.getMat(cv::ACCESS_READ).at<float>(0, 0) << std::endl;
//    std::cout << "Second matrix at 0, 0 has real part " << matComplex2.real.getMat(cv::ACCESS_READ).at<float>(0, 0)
//        << " and imaginary part " << matComplex2.imag.getMat(cv::ACCESS_READ).at<float>(0, 0) << std::endl;
    std::cout << "First matrix is: " << '\n' << mat_complex.toString() << std::endl;
    std::cout << "Second matrix is: " << '\n' << matComplex2.toString() << std::endl;

    //uncomment for division tests
    mat_complex /= matComplex2;
    std::cout << "After division: " << '\n' << mat_complex.toString() << std::endl;

    std::cout << mat_complex.toString() << std::endl;


    cvc::conj(mat_complex);

//    cmshow(mat_complex,"Complex Mat");
    std::cout << "overloading test: " << std::endl;
    std::cout << toMat << std::endl;

    std::cout << "random mat test: " << std::endl;
    std::cout << getMat(5) << std::endl;
    */


}
