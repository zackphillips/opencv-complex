#include <stdio.h>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/ocl.hpp>

#ifndef CMAT_H
#define CMAT_H 1

// cMat class definition
// see: http://stackoverflow.com/questions/4421706/operator-overloading

namespace cvc {

    class cMat
    {
        private:
            // Matrix properties
            int8_t mType;
            cv::Size mSize;
        public:

            // Mat objects
            cv::UMat real;
            cv::UMat imag;

            // Return size of real part as size
            cv::Size size() const {return mSize;};
            void set_size(cv::Size newSize) {mSize = newSize;};

            // Return rows and cols
            uint16_t rows() const {return real.rows;};
            uint16_t cols() const {return real.cols;};

            //Return type in the same way as UMat class
            int8_t type() const {return mType;};
            void set_type(int8_t newType) {mType = newType;};

            // Return as two channel UMat (for DFT)
            cv::UMat getBiChannel()
            {
                cv::Mat tmp;
                cv::Mat complexPlanes[] = {real.getMat(cv::ACCESS_RW), imag.getMat(cv::ACCESS_RW)};
                cv::merge(complexPlanes,2,tmp);
                return tmp.getUMat(cv::ACCESS_RW);
            }

            // Set real and imaginary part as two channel UMat (for DFT)
            void setFromBiChannel(cv::UMat biChannelUMat)
            {
                cv::Mat tmp;
                cv::Mat complexPlanes[] = {cv::Mat::zeros(biChannelUMat.rows, biChannelUMat.cols, biChannelUMat.type()),
                    cv::Mat::zeros(biChannelUMat.rows, biChannelUMat.cols, biChannelUMat.type())};
                cv::split(biChannelUMat.getMat(cv::ACCESS_RW),complexPlanes);
                real = complexPlanes[0].getUMat(cv::ACCESS_RW);
                imag = complexPlanes[1].getUMat(cv::ACCESS_RW);
            }
            // Initialize with nothing
            cMat()
            {
            }

            // Initialize with complex values
            cMat(cv::UMat real_i, cv::UMat imag_i)
            {
                real = real_i.clone();
                imag = imag_i.clone();
                mType = real.type();
                mSize = real.size();
            }

            cMat(cv::Mat real_i, cv::Mat imag_i):cMat(real_i.getUMat(cv::ACCESS_RW), imag_i.getUMat(cv::ACCESS_RW))
            {};


            // Initialize with real values only
            cMat(cv::UMat real_i)
            {
                real = real_i.clone();
                mType = real.type();
                mSize = real.size();
                imag = cv::UMat::zeros(real_i.size(),real_i.type());
            }

            cMat(cv::Mat real_i) : cMat(real_i.getUMat(cv::ACCESS_RW))
            {};

            // Initialize with zeros
            cMat(cv::Size newSize, int8_t newType)
            {
                mType = newType;
                mSize = newSize;
                real = cv::UMat::zeros(mSize, mType);
                imag = cv::UMat::zeros(mSize, mType);
            }

            // Initialize with zeros
            cMat(uint16_t rowCount, uint16_t colCount, int8_t newType)
            {
                mType = newType;
                mSize = cv::Size(rowCount, colCount);
                real = cv::UMat::zeros(mSize, mType);
                imag = cv::UMat::zeros(mSize, mType);
            }

            // Initialize with a given double value
            cMat(cv::Size newSize, const double val)
            {
                mType = CV_64F;
                mSize = newSize;
                cv::multiply(val,cv::UMat::ones(mSize, mType),real);
                imag = cv::UMat::zeros(mSize, mType);
            }

            // Initialize with a given double value
            cMat(uint16_t rowCount, uint16_t colCount, const double val)
            {
                mType = CV_64F;
                mSize = cv::Size(rowCount,colCount);
                cv::multiply(val,cv::UMat::ones(mSize, mType),real);
                imag = cv::UMat::zeros(mSize, mType);
            }

            // Initialize with a given complex value
            cMat(cv::Size newSize, const std::complex<double> val)
            {
                mType = CV_64F;
                mSize = newSize;
                cv::multiply(val.real(),cv::UMat::ones(mSize, mType),real);
                cv::multiply(val.imag(),cv::UMat::ones(mSize, mType),imag);
            }

            // Initialize with a given complex value
            cMat(uint16_t rowCount, uint16_t colCount, const std::complex<double> val)
            {
                mType = CV_64F;
                mSize = cv::Size(rowCount,colCount);
                cv::multiply(val.real(),cv::UMat::ones(mSize, mType),real);
                cv::multiply(val.imag(),cv::UMat::ones(mSize, mType),imag);
            }

            // Helper function for fast assignment

            friend void swap(cMat& first, cMat& second) // nothrow
            {
                // enable ADL (not necessary in our case, but good practice)
                using std::swap;

                // by swapping the members of two objects,
                // the two objects are effectively swapped
                swap(first.real, second.real);
                swap(first.imag, second.imag);
                first.mType = second.type();
                first.mSize = second.size();
            }

            cMat& operator=(cMat other) // (1)
            {
                swap(*this, other); // (2)
                return *this;
            }

            friend bool operator==(const cMat lhs, const cMat rhs){
                cv::UMat diff_r;
                cv::UMat diff_i;
                cv::compare(lhs.real, rhs.real, diff_r, cv::CMP_NE);
                cv::compare(lhs.imag, rhs.imag, diff_i, cv::CMP_NE);
                bool eq_r = cv::countNonZero(diff_r) == 0;
                bool eq_i = cv::countNonZero(diff_i) == 0;
                return (eq_r & eq_i);
            }

            /*******************************************************************
            ************************ ADDITION OPERATORS ************************
            *******************************************************************/

            /*
             * Adds two matrices with the + operator. Does not modify either
             * matrix.
             */
            friend cMat operator+(cMat lhs, const cMat& rhs) {
                cMat output;
                output.set_size(lhs.size());
                output.set_type(lhs.type());
                cv::add(lhs.real, rhs.real, output.real);
                cv::add(lhs.imag, rhs.imag, output.imag);
                return output;
            }

            /*
             * Performs += with two matrices. Modifies *this.
             */
            cMat& operator+=(const cMat& val) {
                *this = *this + val;
                return *this;
            }

            /*
             * Adds a cMat to a double. Adds the double's value to the real
             * part of each matrix component. Does not modify the matrix.
             */
            friend cMat operator+(cMat mat, const double& val) {
                cMat output;
                output.set_size(mat.size());
                output.set_type(mat.type());
                cv::add(mat.real, val, output.real);
                mat.imag.copyTo(output.imag);
                return output;
            }

            /*
             * Performs += with a matrix and a double. Modifies the matrix.
             */
            cMat& operator+=(const double& val) {
                *this = *this + val;
                return *this;
            }

            /*
             * Adds a complex number to each element of a matrix. Does not modify
             * the matrix or the std::complex
             */
            friend cMat operator+(cMat mat, const std::complex<double>& val) {
                cMat output;
                output.set_size(mat.size());
                output.set_type(mat.type());
                cv::add(mat.real, val.real(), output.real);
                cv::add(mat.imag, val.imag(), output.imag);
                return output;
            }

            /*
             * Adds a double to a matrix on the left.
             */
            friend cMat operator+(const std::complex<double>& val, cMat mat) {
                return mat + val;
            }

            /*
             * Performs += with a matrix and an std::complex<double>. Modifies
             * the orignal matrix.
             */
            cMat& operator+=(const std::complex<double>& val) {
                *this = *this + val;
                return *this;
            }

            /*******************************************************************
            ********************** SUBTRACTION OPERATORS ***********************
            *******************************************************************/

            /*
             * Subtracts two matrices. Does not modify either matrix.
             */
            friend cMat operator-(cMat lhs, const cMat& rhs) {
                cMat output;
                output.set_size(lhs.size());
                output.set_type(lhs.type());
                cv::subtract(lhs.real, rhs.real, output.real);
                cv::subtract(lhs.imag, rhs.imag, output.imag);
                return output;
            }

            /*
             * Performs -= on *this. Modifies *this.
             */
            cMat& operator-=(const cMat& val) {
                *this = *this - val;
                return *this;
            }

            /*
             * Subtracts a double from a matrix. Subtracts the double from the
             * real and imaginary part of each element in the matrix.
             */
            friend cMat operator-(cMat mat, const double& val) {
                return mat + (-val);
            }

            /*
             * Subtracts a matrix from a double.
             */
            friend cMat operator-(const double& val, cMat mat) {
                return (-mat) + val;
            }

            /*
             * Performs -= with a matrix and double. Modifies the matrix.
             */
            cMat& operator-=(const double& val) {
                *this = *this - val;
                return *this;
            }

            /*
             * Subtracts a std::complex<double> from each element in a matrix.
             */
            friend cMat operator-(cMat mat, const std::complex<double>& val) {
                return mat + (-val);
            }

            /*
             * Subtracts a double from a matrix on the left.
             */
            friend cMat operator-(const std::complex<double>& val, cMat mat) {
                return val + (-mat);
            }

            /*
             * Performs -= with a matrix and std::complex. Modifies the matrix.
             */
            cMat& operator-=(const std::complex<double>& val) {
                *this = *this - val;
                return *this;
            }

            /*******************************************************************
            ********************* MULTIPLICATION OPERATORS *********************
            *******************************************************************/

            /*
             * Multiplies two matrices. Does not modify either matrix.
             */
            friend cMat operator*(cMat lhs, const cMat& rhs) {
                cMat output;
                cv::UMat tmp1;
                cv::UMat tmp2;

                output.set_size(lhs.size());
                output.set_type(lhs.type());
                cv::multiply(lhs.real, rhs.real, output.real);
                cv::multiply(lhs.imag, rhs.imag, tmp1);
                cv::multiply(lhs.real, rhs.imag, output.imag);
                cv::multiply(lhs.imag, rhs.real, tmp2);
                cv::subtract(output.real, tmp1, output.real);
                cv::add(output.imag, tmp2, output.imag);
                return output;
            }

            /*
             * Performs *= on *this. Modifies *this.
             */
            cMat& operator*=(const cMat& val) {
                *this = *this * val;
                return *this;
            }

            /*
             * Multiplies each element of a matrix with a double. Does not
             * modify the matrix.
             */
            friend cMat operator*(cMat mat, const double& val) {
                cMat output;
                output.set_size(mat.size());
                output.set_type(mat.type());
                cv::multiply(mat.real, val, output.real);
                cv::multiply(mat.imag, val, output.imag);
                return output;
            }

            /*
             * Performs left hand multiplication with a double.
             */
            friend cMat operator*(const double& val, cMat mat) {
                return mat * val;
            }

            /*
             * Performs *= on a matrix with a double.
             */
            cMat& operator *=(const double& val) {
                *this = *this * val;
                return *this;
            }

            /*
             * Performs multiplication with a std::complex<double>. Does not
             * modify the matrix.
             */
            friend cMat operator*(cMat mat, const std::complex<double> z) {
                cMat output;
                cv::UMat tmp1;
                cv::UMat tmp2;

                output.set_size(mat.size());
                output.set_type(mat.type());
                cv::multiply(mat.real, z.real(), output.real);
                cv::multiply(mat.imag, z.imag(), tmp1);
                cv::multiply(mat.real, z.imag(), output.imag);
                cv::multiply(mat.imag, z.real(), tmp2);
                cv::subtract(output.real, tmp1, output.real);
                cv::add(output.imag, tmp2, output.imag);
                return output;
            }

            /*
             * Performs left multiplication with a std::complex<double>
             */
            friend cMat operator*(const std::complex<double> z, cMat mat) {
                return mat * z;
            }

            /*
             * Performs *= with a std::complex
             */
            cMat& operator*=(const std::complex<double> z) {
                *this = *this * z;
                return *this;
            }

            /*
             * Negates each element of a matrix.
             */
            friend cMat operator-(cMat mat) {
                return mat * (-1);
            }

            /*
             * Performs square operation
             */

            friend cMat operator^(cMat mat, const double power) {
              cMat output (mat.size(),mat.type());
              for(int row = 0; row < output.real.rows; row++)
              {
                const double* in_re_row = mat.real.getMat(cv::ACCESS_RW).ptr<double>(row);  // Input
                const double* in_im_row = mat.imag.getMat(cv::ACCESS_RW).ptr<double>(row);  // Input
                double* out_re_row = output.real.getMat(cv::ACCESS_RW).ptr<double>(row);   // Output real
                double* out_im_row = output.imag.getMat(cv::ACCESS_RW).ptr<double>(row);   // Output imag

                for(int col = 0; col < output.real.cols; col++)
                {
                    std::complex<double> z (in_re_row[col],in_im_row[col]);
                    z = std::pow(z,power);
                    out_re_row[col] = z.real();
                    out_im_row[col] = z.imag();
                }
              }
              return output;
            }
            /*******************************************************************
            ************************ DIVISION OPERATORS ************************
            *******************************************************************/

            /*
             * Performs elementwise complex division. Does not modify either
             * matrix.
             */
            friend cMat operator/(cMat lhs, const cMat& rhs) {
                cv::UMat temp;
                cMat conjugate (rhs.real);
                cv::multiply(-1.0, rhs.imag, conjugate.imag);
                cMat output = lhs;
                output *= conjugate;
                cMat divisor = rhs;
                divisor *= conjugate;
                cv::divide(output.real, divisor.real, output.real);
                cv::divide(output.imag, divisor.real, output.imag);
                return output;
            }

            /*
             * Performs /= on this. Modifies *this.
             */
            cMat& operator/=(cMat mat) {
                *this = *this / mat;
                return *this;
            }

            /*
             * Performs double division. Does not modify the matrix.
             */
            friend cMat operator/(cMat mat, const double& val) {
                cv::UMat real;
                cv::UMat im;
                cv::divide(mat.real, val, real);
                cv::divide(mat.imag, val, im);
                return *(new cMat(real, im));
            }

            /*
             * Performs /= on a double. Modifies the matrix.
             */
            cMat& operator/=(const double& val) {
                *this = *this / val;
                return *this;
            }

            /*
             * Performs left division with a double
             */
            friend cMat operator/(const double& val, cMat mat) {
                cv::UMat tmp1;
                cv::UMat tmp2;
                cMat output (mat.real);
                cv::multiply(-1.0, mat.imag, output.imag);
                output *= val;
                cv::multiply(mat.real, mat.real, tmp1);
                cv::multiply(mat.imag, mat.imag, tmp2);
                cv::add(tmp1,tmp2,tmp1);
                cv::divide(output.real, tmp1, output.real);
                cv::divide(output.imag, tmp1, output.imag);
                return output;
            }

            /*
             * Performs left division with a std::complex<double>
             */
            friend cMat operator/(const std::complex<double> z, cMat mat) {
                cv::UMat tmp1;
                cv::UMat tmp2;
                cMat output (mat.real);
                cv::multiply(-1.0, mat.imag, output.imag);
                output *= z;
                cv::multiply(mat.real, mat.real, tmp1);
                cv::multiply(mat.imag, mat.imag, tmp2);
                cv::add(tmp1,tmp2,tmp1);
                cv::divide(output.real, tmp1, output.real);
                cv::divide(output.imag, tmp1, output.imag);
                return output;
            }

            /*
             * Performs complex division with a std::complex. Does not modify the
             * matrix.
             */
            friend cMat operator/(cMat mat, const std::complex<double> z) {
                cMat output = mat * std::conj(z);
                cv::divide(output.real, std::norm(z), output.real);
                cv::divide(output.imag, std::norm(z), output.imag);
                return output;
            }

            /*
             * Performs /= with a std::complex
             */
            cMat& operator/=(const std::complex<double> val) {
                *this = *this / val;
                return *this;
            }

            /*
             * Overloads the << operator to print the matrix.
             */
            friend std::ostream& operator<<(std::ostream& output, const cMat& mat) {
                output << '\n' << mat.toString();
                return output;
            }

            /*
             * Set the element at (m, n) equal to val. If this is not a valid
             * indexing of the matrix, throws an error.
             */
            void set(int m, int n, std::complex<double> val) {
                if (m > this->mSize.height || n > this->mSize.width) {
                    throw std::invalid_argument("invalid matrix index");
                }
                this->real.getMat(cv::ACCESS_RW).at<double>(m, n) = val.real();
                this->imag.getMat(cv::ACCESS_RW).at<double>(m, n) = val.imag();
            }

            /*
             * Gets the element at position (m, n). If this is not a valid
             * index, throws an error.
             */
            std::complex<double>* get(int m, int n) const {
                if (m > this->mSize.height || n > this->mSize.width) {
                    throw std::invalid_argument("invalid matrix index");
                }
                double real = this->real.getMat(cv::ACCESS_READ).at<double>(m, n);
                double im = this->imag.getMat(cv::ACCESS_READ).at<double>(m, n);
                return new std::complex<double>(real, im);
            }

            /*
             * toString method for the cMat class.
             */
            std::string toString() const {
                std::string s = "";
                for (int r = 0; r < this->mSize.height; r++) {
                    s += "[";
                    for (int c = 0; c < this->mSize.width; c++) {
                        std::complex<double>* z = this->get(r, c);
                        std::string entry = std::to_string(z->real()) + " + " + std::to_string(z->imag()) + "*j";
                        if (c != 0) {
                            s += ", ";
                        }
                        s += entry;
                    }
                    s += "]";
                    s += '\n';
                }
                return s;
            }

            /*
             * Gets the size of the matrix.
             */
            cv::Size getSize() {
                return this->mSize;
            }

            /*
             * Changes the rth row of A to realRow + j * imRow.
             */
            void setRow(cv::Mat realRow, cv::Mat imRow, int r) {
                realRow.row(0).copyTo(this->real.getMat(cv::ACCESS_RW).row(r));
                imRow.row(0).copyTo(this->imag.getMat(cv::ACCESS_RW).row(r));
            }

            /*
             * Sets the rth row of A to row.
             */
            void setRow(cMat row, int r) {
                this->setRow(row.real.getMat(cv::ACCESS_RW), row.imag.getMat(cv::ACCESS_RW), r);
            }

            /*
             * Gets the real part of the rth row.
             */
            cv::Mat getRealRow(int r) {
                if (r < 0 || r > this->mSize.height) {
                    throw std::invalid_argument("row is out of bounds.");
                }
                return this->real.getMat(cv::ACCESS_RW).row(r);
            }

            /*
             * Gets the imaginary part of the rth row.
             */
            cv::Mat getImagRow(int r) {
                if (r < 0 || r > this->mSize.height) {
                    throw std::invalid_argument("row is out of bounds.");
                }
                return this->imag.getMat(cv::ACCESS_RW).row(r);
            }

            /*
             * Changes the cth column of A to realCol + j * imCol.
             */
            void setCol(cv::Mat realCol, cv::Mat imCol, int c) {
                realCol.col(0).copyTo(this->real.getMat(cv::ACCESS_RW).col(c));
                imCol.col(0).copyTo(this->imag.getMat(cv::ACCESS_RW).col(c));
            }

            /*
             * Gets the real part of the cth column of A.
             */
            cv::Mat getRealCol(int c) {
                if (c < 0 || c > this->mSize.width) {
                    throw std::invalid_argument("column is out of bounds.");
                }
                return this->real.getMat(cv::ACCESS_RW).col(c);
            }

            /*
             * Gets the imaginary part of the cth column of A.
             */
            cv::Mat getImagCol(int c) {
                if (c < 0 || c > this->mSize.width) {
                    throw std::invalid_argument("column is out of bounds.");
                }
                return this->imag.getMat(cv::ACCESS_RW).col(c);
            }

            /*
             * Hard copy
             */
            cMat copy()
            {
                return *new cMat(this->real,this->imag);
            }

            /*
             * Transpose operator.
             */
            cMat t() {
                return *new cMat(this->real.t(),this->imag.t());
            }
            /*
             * Hermitian operator.
             */
            cMat h() {
                cv::UMat temp;
                cv::multiply(-1,this->imag.t(),temp);
                return *new cMat(this->real.t(),temp);
            }

        };

    void printOclPlatformInfo();

    // Display functions
    void cmshow(cMat matToShow, std::string windowTitle);
    void mouseCallback_cmshow(int event, int x, int y, int, void* param);

    cMat abs(const cMat& inMat);
    cMat angle(const cMat& inMat);
    cMat conj(const cMat& inMat);
    cMat exp(const cMat& inMat);
    cMat log(const cvc::cMat& inMat);
    cMat vec(const cvc::cMat& inMat);
    cMat reshape(const cvc::cMat& inMat, const int rows);
    cMat fft2(cvc::cMat& inMat);
    cMat ifft2(cvc::cMat& inMat);
    void fftshift(cvc::cMat& input, cvc::cMat& output);
    void ifftshift(cvc::cMat& input, cvc::cMat& output);
    void circularShift(cvc::cMat& input, cvc::cMat& output, int16_t x, int16_t y);
}

#endif
