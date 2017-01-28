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

class cMat: public cv::UMat
{
    private:
        cv::Size sz;
    public:
        cv::UMat real;
        cv::UMat imag;

        cv::Size size() {return sz;};

        // Initialize with complex values
        cMat(cv::UMat real_i, cv::UMat imag_i)
        {
            sz = real_i.size();
            real = real_i;
            imag = imag_i;
        }
        cMat(cv::Mat real_i, cv::Mat imag_i)
            {cMat(real_i.getUMat(cv::ACCESS_RW), imag_i.getUMat(cv::ACCESS_RW));};
        // Initialize with real values only
        cMat(cv::UMat real_i)
        {
            sz = real_i.size();
            real = real_i;
            imag = cv::UMat::zeros(real_i.size(),real_i.type());
        }
        cMat(cv::Mat real_i)
            {cMat(real_i.getUMat(cv::ACCESS_RW));};

        // Initialize with zeros
        cMat(cv::Size newSz, int8_t type)
        {
            sz = newSz;
            real = cv::UMat::zeros(sz, type);
            imag = cv::UMat::zeros(sz, type);
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

        // Matrix element-wise addition
        cMat& operator+=(const cMat& val)
        {
            cMat result(this->size(),this->type());
            cv::add(val, *this, *this);
            return *this;
        }
        // Scalar element-wise addition
        cMat& operator+=(const double& val)
        {
            cMat result(this->size(),this->type());
            cv::add(val, *this, *this);
            return *this;
        }

        // Matrix element-wise subtraction
        cMat& operator-=(const cMat& val)
        {

            cMat result(this->size(),this->type());
            cv::subtract(val, *this, *this);
            return *this;
        }
        // Scaler element-wise subtraction
        cMat& operator-=(const double& val)
        {
            cMat result(this->size(),this->type());
            cv::subtract(val, *this, *this);
            return *this;
        }

        // Matrix element-wise multiplication
        cMat& operator*=(const cMat& val)
        {
            cMat result(this->size(),this->type());
            cv::multiply(val, *this, *this);
            return *this;
        }
        // Scaler element-wise multiplication
        cMat& operator*=(const double& val)
        {
            cMat result(this->size(),this->type());
            cv::multiply(val, *this, *this);
            return *this;
        }

        // Matrix element-wise division
        cMat& operator/=(const cMat& val)
        {
            cMat result(this->size(),this->type());
            cv::divide(val, *this, *this);
            return *this;
        }
        // Scaler element-wise division
        cMat& operator/=(const double& val)
        {
            cMat result(this->size(),this->type());
            cv::divide(val, *this, *this);
            return *this;
        }

};

void printOclPlatformInfo();

// Display functions
void cmshow(cMat matToShow, std::string windowTitle);
void mouseCallback_cmshow(int event, int x, int y, int, void* param);


#endif
