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
    cMat noRef = *A;
    std::cout << "Dereferenced A is: " << noRef << std::endl;
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

// return a simple matrix to test
cMat getSimpleMat(int size, double stagger) {
    cMat A = *getMat(size);
    for (int r = 0; r < size; r++) {
        for (int c = 0; c < size; c++) {
            std::complex<double> z (r + 2*c + stagger, 3*r + c - stagger);
            A.set(r, c, z);
        }
    }
    return A;
}

// add test matrices
void testAddMats() {
    std::cout << "Testing Matrix Addition." << std::endl;
    int size = 5;
    cMat A = *getMat(size);
    cMat B = *getMat(size);
    for (int r = 0; r < size; r++) {
        for (int c = 0; c < size; c++) {
            std::complex<double> z1 (r, 2*c);
            std::complex<double> z2 (3*r, c);
            A.set(r, c, z1);
            B.set(r, c, z2);
        }
    }
    std::cout << "A is: " << '\n' << A << std::endl;
    std::cout << "B is: " << '\n' << B << std::endl;
    cMat C = A + B;
    std::cout << "C := A + B is: " << '\n' << C << std::endl;
    C += B;
    std::cout << "C += B is: " << '\n' << C << std::endl;
}

// add double to cMat
void testAddDouble() {
    std::cout << "Testing Double Addition." << std::endl;
    cMat A = getSimpleMat(5, 2.0);
    std::cout << "A is: \n" << A << std::endl;
    cMat B = A + 6.0;
    std::cout << "B := A + 6.0 is: \n" << B << std::endl;
    B += 3.0;
    std::cout << "B += 3.0 is: \n" << B << std::endl;
}

// add std::complex to cMat
void testAddZ() {
    std::cout << "Testing std::complex Addition." << std::endl;
    cMat A = getSimpleMat(4, 4.0);
    std::cout << "A is: \n" << A << std::endl;
    std::complex<double> z1 (3.0, 7.0);
    std::complex<double> z2 (1.0, -1.0);
    std::complex<double> z3 (2.0, 1.0);
    cMat B = A + z1;
    std::cout << "B := A + (3 + 7i) is: \n" << B << std::endl;
    cMat C = z3 + A;
    std::cout << "C := (2 + i) + A is: \n" << C << std::endl;
    B += z2;
    std::cout << "B += (1 - i) is: " << B << std::endl;
}

// test subtract matrices
void testSubtractMats() {
    std::cout << "Testing Matrix Subtraction." << std::endl;
    int size = 5;
    cMat A = *getMat(size);
    cMat B = *getMat(size);
    for (int r = 0; r < size; r++) {
        for (int c = 0; c < size; c++) {
            std::complex<double> z1 (5*r, -2*c);
            std::complex<double> z2 (2*r, -4 + c);
            A.set(r, c, z1);
            B.set(r, c, z2);
        }
    }
    std::cout << "A is: " << '\n' << A << std::endl;
    std::cout << "B is: " << '\n' << B << std::endl;
    cMat C = A - B;
    std::cout << "C := A - B is: " << '\n' << C << std::endl;
    C -= A;
    std::cout << "C -= A is: " << '\n' << C << std::endl;
}

// test subtract double
void testSubtractDouble() {
    std::cout << "Testing Double Subtraction." << std::endl;
    cMat A = getSimpleMat(5, 1.0);
    std::cout << "A is: \n" << A << std::endl;
    cMat B = A - 4.0;
    std::cout << "B := A - 4.0 is: \n" << B << std::endl;
    cMat C = 7.0 - A;
    std::cout << "C := 7.0 - A is: \n" << C << std::endl;
    A -= 2.0;
    std::cout << "A -= 2.0 is: \n" << A << std::endl;
}

//test subtract std::complex
void testSubtractZ() {
    std::cout << "Testing std::complex Subtraction." << std::endl;
    cMat A = getSimpleMat(5, 3.0);
    std::cout << "A is: \n" << A << std::endl;
    std::complex<double> z1 (2, 8);
    std::complex<double> z2 (3, -1);
    std::complex<double> z3 (2, 5);
    cMat B = A - z1;
    std::cout << "B := A - (2 + 8i) is: \n" << B << std::endl;
    cMat C = z2 - A;
    std::cout << "C := (3 - i) - A is: \n" << C << std::endl;
    A -= z3;
    std::cout << "A -= (2 + 5i) is: \n" << A << std::endl;
}

// test multiply matrices pointwise
void testMultMats() {
    std::cout << "Testing Matrix Multiplication." << std::endl;
    cMat A = getSimpleMat(3, 6.0);
    cMat B = getSimpleMat(3, 2.0);
    std::cout << "A is: \n" << A << std::endl;
    std::cout << "B is: \n" << B << std::endl;
    cMat C = A * B;
    std::cout << "C := A * B is: \n" << C << std::endl;
    A *= B;
    std::cout << "A *= B is: \n" << A << std::endl;
}

// test multiply matrix by double
void testMultDouble() {
    std::cout << "Testing Double Multiplication." << std::endl;
    cMat A = getSimpleMat(5, 2.0);
    std::cout << "A is: \n" << A << std::endl;
    cMat B = A * 3.0;
    std::cout << "B := A * 3.0 is: \n" << B << std::endl;
    cMat C = -3.0 * A;
    std::cout << "C := -3.0 * A is: \n" << C << std::endl;
    A *= 10.0;
    std::cout << "A *= 10.0 is: \n" << A << std::endl;
}

void testMultZ() {
    std::cout << "Testing std::complex Multiplication." << std::endl;
    cMat A = getSimpleMat(5, 5.0);
    std::complex<double> z1 (2, 4);
    std::complex<double> z2 (1, -2);
    std::complex<double> z3 (3, 8);
    std::cout << "A is: \n" << A << std::endl;
    cMat B = A * z1;
    std::cout << "B := A * (2 + 4i) is: \n" << B << std::endl;
    cMat C = z2 * A;
    std::cout << "C := (1 - 2i) * A is: \n" << C << std::endl;
    A *= z3;
    std::cout << "A *= (3 + 8i) is: \n" << A << std::endl;
}

void testNegate() {
    std::cout << "Testing Negation." << std::endl;
    cMat A = getSimpleMat(3, 2.0);
    std::cout << "A is: \n" << A << std::endl;
    cMat B = -A;
    std::cout << "B := -A is: \n" << B << std::endl;
}

// test divide matrices pointwise
void testDivMats() {
    //TODO test division by 0 error
    std::cout << "Testing Matrix Division." << std::endl;
    cMat A = getSimpleMat(3, 2.0);
    cMat B = getSimpleMat(3, 1.0);
    std::cout << "A is: \n" << A << std::endl;
    std::cout << "B is: \n" << B << std::endl;
    cMat C = A / B;
    std::cout << "C := A / B is: \n" << C << std::endl;
    A /= B;
    std::cout << "A /= B is: \n" << A << std::endl;
}

// test divide matrix by double
void testDivDouble() {
    std::cout << "Testing Double Division." << std::endl;
    int size = 4;
    cMat A = getSimpleMat(5, 2.0);
    std::cout << "A is: \n" << A << std::endl;
    cMat B = A / 5.0;
    std::cout << "B := A / 5.0 is: \n" << B << std::endl;
    A /= 10.0;
    std::cout << "A /= 10.0 is: \n" << A << std::endl;
}

// test divide matrix by std::complex
void testDivZ() {
    std::cout << "Testing Complex Division." << std::endl;
    cMat A = getSimpleMat(5, -2.0);
    std::cout << "A is: \n" << A << std::endl;
    std::complex<double> z1 (3, 1);
    std::complex<double> z2 (0, 1);
    cMat B = A / z1;
    std::cout << "B := A / (3 + i) is: \n" << B << std::endl;
    A /= z2;
    std::cout << "A /= i is: \n" << A << std::endl;
}

// test setRow
void testSetRowCol() {
    std::cout << "Testing Set Row." << std::endl;
    cMat A = getSimpleMat(5, 1);
    std::cout << "A is: \n" << A << std::endl;
    cv::Mat realRow (1, 5, 5);
    cv::Mat imRow (1, 5, 5);
    for (int i = 0; i < 5; i++) {
        realRow.at<float>(0, i) = i;
        imRow.at<float>(0, i) = i;
    }
    std::cout << "realRow is: \n" << realRow << std::endl;
    A.setRow(realRow, imRow, 2);
    std::cout << "With new rows, A is: \n" << A << std::endl;

    cv::Mat realCol (5, 1, 5);
    cv::Mat imCol(5, 1, 5);
    for (int i = 0; i < 5; i++) {
        realCol.at<float>(i, 0) = 5*i;
        imCol.at<float>(i, 0) = 5*i;
    }
    A.setCol(realCol, imCol, 0);
    std::cout << "With new rows, C is: \n" << A << std::endl;
}

//test getRow
void testGetRow() {
    std::cout << "Testing Get Row." << std::endl;
    cMat A = getSimpleMat(5, 0.0);
    std::cout << "A is: \n" << A << std::endl;
    cv::Mat realRow = A.getRealRow(1);
    cv::Mat imagRow = A.getImagRow(1);
    std::cout << "Row 1 of A has real part: \n" << realRow << std::endl;
    std::cout << "Row 1 of A has imaginary part: \n" << imagRow << std::endl;
}

// test transpose
void testTranspose() {
    std::cout << "Testing Transpose." << std::endl;
    cMat A = *(getMat(1, 5, 0.0, 1.0));
    std::cout << "A is: \n" << A << std::endl;
    cMat AT = A.t();
    std::cout << "A^T is: \n" << AT << std::endl;
    std::cout << "A^T^T is: \n" << AT.t() << std::endl;
}

// test 1D FFT.
void testFFT() {

}

// test 2D FFT.
void testFFT2() {

}

int main(int argc, char** argv ){

    // testToString();
    // testGet();
    // testSet();
    // testAddMats();
    // testAddDouble();
    // testAddZ();
    // testSubtractMats();
    // testSubtractDouble();
    // testSubtractZ();
    // testMultMats();
    // testMultDouble();
    // testMultZ();
    // testNegate();
    // testDivMats();
    // testDivDouble();
    // testDivZ();
    // testSetRowCol();
    // testGetRow();
    testTranspose();

}
