#include <stdio.h>
#include <cmath>
#include <iostream>

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
          cv::Size size() {return mSize;};

          //Return type in the same way as UMat class
          int8_t type() {return mType;};

          // Initialize with complex values
          cMat(cv::UMat real_i, cv::UMat imag_i)
          {
              real = real_i.clone();
              imag = imag_i.clone();
              mType = real.type();
              mSize = real.size();
          }

          cMat(cv::Mat real_i, cv::Mat imag_i)
              {cMat(real_i.getUMat(cv::ACCESS_RW), imag_i.getUMat(cv::ACCESS_RW));};


          // Initialize with real values only
          cMat(cv::UMat real_i)
          {
              real = real_i.clone();
              mType = real.type();
              mSize = real.size();
              imag = cv::UMat::zeros(real_i.size(),real_i.type());
          }
          cMat(cv::Mat real_i)
              {cMat(real_i.getUMat(cv::ACCESS_RW));};

          // Initialize with zeros
          cMat(cv::Size newSize, int8_t newType)
          {
              mType = newType;
              mSize = newSize;
              real = cv::UMat::zeros(mSize, mType);
              imag = cv::UMat::zeros(mSize, mType);
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
          }

          cMat& operator=(cMat other) // (1)
          {
              swap(*this, other); // (2)
              return *this;
          }

          // Arithmatic Operators

          // Matrix element-wise subtraction
          cMat& operator+=(const cMat& val)
          {
              cv::add(val.real, this->real, this->real);
              cv::add(val.real, this->imag, this->imag);
              return *this;
          }

          // Scaler element-wise subtraction
          cMat& operator+=(const double& val)
          {
              cv::add(val, this->real, this->real);
              cv::add(val, this->imag, this->imag);
              return *this;
          }

          // Matrix element-wise subtraction
          cMat& operator-=(const cMat& val)
          {
              cv::subtract(val.real, this->real, this->real);
              cv::subtract(val.real, this->imag, this->imag);
              return *this;
          }

          // Scaler element-wise subtraction
          cMat& operator-=(const double& val)
          {
              cv::subtract(val, this->real, this->real);
              cv::subtract(val, this->imag, this->imag);
              return *this;
          }

          // Matrix element-wise multiplication
          cMat& operator*=(const cMat& val)
          {
              cv::UMat tmp1;
              cv::UMat tmp2;

              cv::multiply(this->real, val.real, tmp1);
              cv::multiply(this->imag, val.imag, tmp2);

              cv::subtract(tmp1, tmp2, this->real);
              cv::add(tmp1, tmp2, this->imag);

              return *this;
          }


          // Scaler element-wise multiplication
          cMat& operator*=(const double& val)
          {
              cv::multiply(val, this->real, this->real);
              cv::multiply(val, this->imag, this->imag);
              return *this;
          }

          /*
          // Matrix element-wise division
          cMat& operator/=(const cMat& mat)
          {
              cMat result(size, type);
              cv::divide(val, *this, *this);
              return *this;
          }
          */

          // Scaler element-wise division
          cMat& operator/=(const double& val)
          {
              cv::divide(val, this->real, this->real);
              cv::divide(val, this->imag, this->imag);
              return *this;
          }

  };

  void printOclPlatformInfo();

  // Display functions
  void cmshow(cMat matToShow, std::string windowTitle);
  void mouseCallback_cmshow(int event, int x, int y, int, void* param);

  cMat abs(cMat& inMat);
  cMat angle(cMat& inMat);
  cMat conj(cMat& inMat);
}



#endif
