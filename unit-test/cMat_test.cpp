#include "../cMat.h"
#include <stdio.h>
#include <complex>
#include <cmath>
#include <iostream>
#include <time.h>

using namespace cvc;

cMat* getMat(int m, int n, double low, double high) {
    cv::UMat real(cv::Size(m, n), CV_64F);
    cv::UMat im(cv::Size(m, n), CV_64F);
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

//test CMatOnes

void testCMatOnes(){
    std::complex<double> z (1,0);
    cMat A (cv::Size(3,3),z);
    std::cout << "A is: " << '\n' << A << '\n' << "type = " << std::to_string(A.type()) << '\n' << std::endl;
    cMat B (3,3,z);
    std::cout << "B is: " << '\n' << B << '\n' << "type = " << std::to_string(B.type()) << '\n' << std::endl;
}

//test toString. 4x2 matrix.
void testToString() {
    std::cout << "Testing toString." << std::endl;
    double low = 0.0;
    double high = 1.0;
    cv::UMat toMatR(cv::Size(2, 4), CV_64F);
    cv::UMat toMatI(cv::Size(2, 4), CV_64F);
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
                " + " << z->imag() << "*j" << std::endl;
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

// test divide double by matrix
void testDoubleDiv() {
    std::cout << "Testing Double Division." << std::endl;
    cMat A = getSimpleMat(5, -2.0);
    std::cout << "A is: \n" << A << std::endl;
    cMat B = 5.0 / A;
    std::cout << "B := 5 / A is: \n" << B << std::endl;
}

// test divide std::complex by matrix
void testZDiv() {
    std::cout << "Testing Complex Division." << std::endl;
    cMat A = getSimpleMat(5, -2.0);
    std::cout << "A is: \n" << A << std::endl;
    std::complex<double> z1 (3, 1);
    cMat B = z1 / A;
    std::cout << "B := (3 + i) / A is: \n" << B << std::endl;
}

// test power
void testpower(){
    cMat A (5,5,std::complex<double> (3,1));
    std::cout << "A is: \n" << A << std::endl;
    cMat Asquare = A^2;
    std::cout << "A^2 is: \n" << Asquare << std::endl;
    cMat Acube = A^3;
    std::cout << "A^3 is: \n" << Acube << std::endl;
    std::cout << "A is: \n" << A << std::endl;
    double val = 5;
    cMat B (5,5,val);
    std::cout << "B is: \n" << B << std::endl;
    cMat Bto1 = B^1;
    std::cout << "B^1 is: \n" << Bto1 << std::endl;
    cMat Bsquare = B^2;
    std::cout << "B^2 is: \n" << Bsquare << std::endl;
    std::cout << "B is: \n" << B << std::endl;
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

// test Hermitian
void testHermitian() {
    std::cout << "Testing Hermitian." << std::endl;
    cMat A = *(getMat(1, 5, 0.0, 1.0));
    std::cout << "A is: \n" << A << std::endl;
    std::cout << "A^H is: \n" << A.h() << std::endl;
    std::cout << "A^H^H is: \n" << A.h().h()<< std::endl;
}

// Test 2D fftshift
void testFFTShift() {
    std::cout << "Testing fftshift" << std::endl;
    cMat A = getSimpleMat(5, 0.0);
    std::cout << "A is: \n" << A << std::endl;
    fftshift(A, A);
    std::cout << "fftshift(A) is: \n" << A << std::endl;
}

// Test 2D ifftshift
void testIFFTShift() {
    std::cout << "Testing ifftshift" << std::endl;
    cMat A = getSimpleMat(5, 0.0);
    std::cout << "A is: \n" << A << std::endl;
    ifftshift(A, A);
    std::cout << "ifftshift(A) is: \n" << A << std::endl;
}

// Test 2D FFT.
void testFFT2() {
    std::cout << "Testing FFT" << std::endl;
    cMat A = getSimpleMat(5, 0.0);
    std::cout << "A is: \n" << A << std::endl;
    cMat A_ft = fft2(A);
    std::cout << "A_ft is: \n" << A_ft << std::endl;
}

// Test 2D iFFT.
void testIFFT2() {
    std::cout << "Testing FFT" << std::endl;
    cMat A = getSimpleMat(5, 0.0);
    std::cout << "A is: \n" << A << std::endl;
    cMat A_ift = ifft2(A);
    std::cout << "A_ift is: \n" << A_ift << std::endl;
}

// Test FFT.
void testFFT() {
    std::cout << "Testing FFT" << std::endl;
    cMat A = getSimpleMat(6, 0.0);
    std::cout << "A is: \n" << A << std::endl;
    cMat A_ft_ift = A;
    for (int fft_num=0;fft_num<1;fft_num++){
        A_ft_ift = fft2(A_ft_ift);
        A_ft_ift = ifft2(A_ft_ift);
    }
    std::cout << "A_ft_ift is: \n" << A_ft_ift << std::endl;
    cv::UMat diff;
    cv::compare(abs(A_ft_ift).real,abs(A).real,diff,cv::CMP_NE);
    int8_t nz = cv::countNonZero(diff);
    std::cout << "nz = \n" << std::to_string(nz)<<"\n" << std::endl;
    bool eq = (nz == 0);
    //bool eq = (A_ft_ift == A);
    std::cout << "A_ft_ift == A is: \n" << eq << std::endl;

}

// Test FFTSHIFT.
void testShift() {
    cMat A = getSimpleMat(5, 0.0);
    std::cout << "A is: \n" << A << std::endl;
    cMat A_shift = A;
    ifftshift(A_shift,A_shift);
    fftshift(A_shift,A_shift);
    std::cout << "A_shift is: \n" << A_shift << std::endl;
    bool eq = (A == A_shift);
    std::cout << "A_shift == A is: \n" <<  std::boolalpha << eq << std::endl;
}

// Test cmshow
void testcmshow(){
    cMat A = getSimpleMat(5, 0.0);
    std::cout << "A is: \n" << A << std::endl;

    // real part
    cMat Ar_disp (A.real);
    Ar_disp /= 12; Ar_disp *= 255;
    std::cout << "Ar_disp is: \n" << Ar_disp << std::endl;

    // imaginary part
    cMat Ai_disp (A.imag);
    Ai_disp /= 16; Ai_disp *= 255;
    std::cout << "Ai_disp is: \n" << Ai_disp << std::endl;

    // absolute value
    double Aabs_min, Aabs_max;
    cMat Aabs_disp = abs(A);
    cv::minMaxLoc(Aabs_disp.real, &Aabs_min, &Aabs_max);
    Aabs_disp = (Aabs_disp - Aabs_min)/(Aabs_max - Aabs_min);
    cv::minMaxLoc(Aabs_disp.real, &Aabs_min, &Aabs_max);
    Aabs_disp *= 255; Aabs_disp /=Aabs_max;
    std::cout << "Aabs_disp is: \n" << Aabs_disp << std::endl;

    // phase value
    double Aangle_min, Aangle_max;
    cMat Aangle_disp = angle(A);
    cv::minMaxLoc(Aangle_disp.real, &Aangle_min, &Aangle_max);
    Aangle_disp = (Aangle_disp - Aangle_min)/(Aangle_max - Aangle_min);
    cv::minMaxLoc(Aangle_disp.real, &Aangle_min, &Aangle_max);
    Aangle_disp *= 255; Aangle_disp /=Aangle_max;
    std::cout << "Aangle_disp is: \n" << Aangle_disp << std::endl;

    // see if A changes
    std::cout << "A is: \n" << A << std::endl;

    cmshow(A,"matrix A");
}

// test complex operation
void test(){
    cMat A = getSimpleMat(5, 0.0);
    std::cout << "A is: \n" << A << std::endl;
    cMat Aabs = abs(A);
    std::cout << "abs(A) is: \n" << Aabs << std::endl;
    cMat Aangle = angle(A);
    std::cout << "angle(A) is: \n" << Aangle << std::endl;
    cMat Aconj = conj(A);
    std::cout << "conj(A) is: \n" << Aconj << std::endl;
    cMat Aexp = exp(A);
    std::cout << "exp(A) is: \n" << Aexp << std::endl;
    std::cout << "A is: \n" << A << std::endl;
    cMat Alog = log(Aexp);
    std::cout << "log(A) is: \n" << Alog << std::endl;
    std::cout << "A is: \n" << A << std::endl;
    cMat Avec = vec(A);
    std::cout << "vec(A) is: \n" << Avec << std::endl;
    std::cout << "A is: \n" << A << std::endl;
    cMat Areshape = reshape(Avec,A.real.rows);
    std::cout << "reshape(A,rows) is: \n" << Areshape << std::endl;
    std::cout << "A is: \n" << A << std::endl;
}

// Test sum
void testSum()
{
   cMat A = *getMat(3, 2);
   std::cout << "A is: \n" << A << std::endl;
   std::cout <<"Shape of A :"<< A.shape()<<std::endl;
   std::cout <<"Size of A :"<< A.size()<<std::endl;

   std::complex<double> z = sum(A);
   printf("sum(A) is (%3.2f,%3.2f)\n",z.real(),z.imag());
   cMat Asum_rows = sum(A,0);
   cMat Asum_cols = sum(A,1);
   std::cout << "sum(A,0) = "<< Asum_rows << std::endl;
   std::cout <<"Shape of sum(A,0) :"<< Asum_rows.shape()<<std::endl;
   std::cout <<"Size of sum(A,0) :"<< Asum_rows.size()<<std::endl;
   std::cout << "sum(A,1) = "<< Asum_cols << std::endl;
   std::cout <<"Shape of sum(A,1) :"<< Asum_cols.shape()<<std::endl;
   std::cout <<"Size of sum(A,1) :"<< Asum_cols.size()<<std::endl;
}

// Test max
void testMinMax()
{
    cMat A = getSimpleMat(3, 0.0);
    std::cout<<"A is "<<A<<std::endl;
    std::cout<<"max(A) is "<<std::to_string(max(A))<<std::endl;
    std::cout<<"max(A,0) is "<<max(A,0)<<std::endl;
    std::cout<<"max(A,1) is "<<max(A,1)<<std::endl;
    std::cout<<"min(A) is "<<std::to_string(min(A))<<std::endl;
    std::cout<<"min(A,0) is "<<min(A,0)<<std::endl;
    std::cout<<"min(A,1) is "<<min(A,1)<<std::endl;
}

// Test norm
void testNorm()
{
    cMat A = *getMat(3, 1);
    std::cout<<"A is "<<A<<std::endl;
    std::cout<<"abs(A) is "<<abs(A)<<std::endl;
    std::cout<<"norm(A) is "<<std::to_string(norm(A))<<std::endl;
    std::cout<<"norm(A,0) is "<<std::to_string(norm(A,0))<<std::endl;
    std::cout<<"norm(A,1) is "<<std::to_string(norm(A,1))<<std::endl;
    std::cout<<"norm(A,2) is "<<std::to_string(norm(A,2))<<std::endl;
}

// Test real.
void testRealImag() {
    cMat A = getSimpleMat(5, 3.0);
    std::cout << "A is: " << A << std::endl;
    std::cout << "Real part of A is: " << real(A) << std::endl;
    std::cout << "Imaginary part of A is: " << imag(A) << std::endl;
}

// Test meshgrid.
void testMeshgrid() {
    cMat x = *getMat(4, 1);
    cMat y = *getMat(1, 4);
    std::cout << "x is: " << x << std::endl;
    std::cout << "y is: " << y << std::endl;

    std::vector<cMat> lst = meshgrid(x, y);
    std::cout << "0th component (X) of meshgrid: " << lst[0] << std::endl;
    std::cout << "1st component (Y) of meshgrid: " << lst[1] << std::endl;
}

void testMultiDim() {
    int * size = new int[4];
    size[0] = 5;
    size[1] = 3;
    size[2] = 4;
    size[3] = 1;
    cMat A = zeros(4, size);
    std::cout << "A is: " << A << std::endl;
}

void testOldEllipse() {
    cv::Point2f center (300, 500);
    cv::Size2f size (200, 400);
    //2 x 4 rectangle centered at 3, 3?
    cv::RotatedRect rect (center, size, 0.0);
    cv::Point2f pts [4];
    rect.points(pts);
    for (int i = 0; i < 4; i++) {
        std::cout << pts[i] << std::endl;
    }
    cv::Mat img = cv::Mat::zeros(1024, 1024, CV_64F);
    cv::ellipse(img, center, size, 0.0, 0.0, 360.0, cv::Scalar(255, 0, 0));
    cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE);
    cv::imshow("Ellipse", img);
    cv::waitKey(0);
}

void testEllipse() {
    cMat ell = zeros(1024, 1024);
    cv::Point2f center = * new cv::Point2f(300.0, 500.0);
    cv::Size2f size = * new cv::Size2f(200.0, 400.0);
    cvc::ellipse(ell, center, size, 0.0, 0.0, 360.0, cv::Scalar(255, 0, 0));
    //TODO ask about errors with cmshow
    cvc::cmshow(ell, "Ellipse");
}

int main(int argc, char** argv){

    // testCMatOnes();
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
    // testDoubleDiv();
    // testZDiv();
    // testpower();
    // testSetRowCol();
    // testGetRow();
    // testTranspose();
    // testHermitian();
    // testFFTShift();
    // testIFFTShift();
    // testFFT2();
    // testIFFT2();
    // testFFT();
    // testShift();
    // testcmshow();
    // test();
    // testSum();
    // testMinMax();
    // testNorm();
    // testRealImag();
    // testMeshgrid();
    // testMultiDim();
    // testOldEllipse();
    testEllipse();
    /*clock_t t1,t2;
    //cMat A = *getMat(3);
    cMat A (3,3,std::complex<double> (2.0,1.0));
    t1 = clock();

    for(int runtest= 0; runtest<1;runtest++){
      cMat B = A^2;
      //std::cout<<"A is :\n"<<A<<std::endl;
      std::cout<<"B is :\n"<<B<<std::endl;
      //B = reshape(B,A.real.rows);
      //std::cout<<"B is :\n"<<B<<std::endl;
      //B.set(0,0,std::complex<double> (3,2));
      //std::cout<<"A is :\n"<<A<<std::endl;
      //std::cout<<"B is :\n"<<B<<std::endl;
    }
    t2 = clock();
    printf("run time = %f second.",(double)(t2-t1)/CLOCKS_PER_SEC);*/
}
