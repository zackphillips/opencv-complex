#include "cMat.h"
#include <stdio.h>
#include <cmath>
#include <iostream>


inline cvc::cMat operator+(cvc::cMat lhs, const cvc::cMat& rhs) { lhs += rhs; return lhs; }
inline cvc::cMat operator+(cvc::cMat lhs, const double& rhs) { lhs += rhs; return lhs; }

inline cvc::cMat operator-(cvc::cMat lhs, const cvc::cMat& rhs) { lhs -= rhs; return lhs; }
inline cvc::cMat operator-(cvc::cMat lhs, const double& rhs) { lhs -= rhs; return lhs; }

inline cvc::cMat operator*(cvc::cMat lhs, const cvc::cMat& rhs) { lhs *= rhs; return lhs; }
inline cvc::cMat operator*(cvc::cMat lhs, const double& rhs) { lhs *= rhs; return lhs; }

//inline cMat operator/(cMat lhs, const cMat& rhs) { lhs /= rhs; return lhs; }
inline cvc::cMat operator/(cvc::cMat lhs, const double& rhs) { lhs /= rhs; return lhs; }

void printOclPlatformInfo()
{
  //OpenCV: Platform Info
  std::vector<cv::ocl::PlatformInfo> platforms;
  cv::ocl::getPlatfomsInfo(platforms);

  // FOR CPU - run this in terminal
  // export OPENCV_OPENCL_DEVICE=":CPU:0"

  // FOR GPU - run this in terminal
  // export OPENCV_OPENCL_DEVICE=":GPU:0"

  //OpenCV Platforms
  std::cout << "OpenCL Platforms:" <<std::endl;
  for (size_t i = 0; i < platforms.size(); i++)
  {
      const cv::ocl::PlatformInfo* platform = &platforms[i];

          //Platform Name
      std::cout << "Platform Name: " << platform->name().c_str() << "\n";

          //Access known device
      cv::ocl::Device current_device;

      for (int j = 0; j < platform->deviceNumber(); j++)
      {
          //Access Device
          platform->getDevice(current_device, j);
          std::cout << "Device Name: " << current_device.name().c_str() << "\n";
          std::cout << "Available: " << current_device.available() << "\n";
          std::cout << "Image Support: " << current_device.imageSupport()<< "\n";
          std::cout << "OpenCL Version: " << current_device.OpenCL_C_Version().c_str() << "\n";

      }
  }
}

void cvc::cmshow(cvc::cMat matToShow, std::string windowTitle)
{
    cv::UMat realMat;
    cv::UMat imagMat;
    cv::UMat ampMat;
    cv::UMat phaseMat;

    std::cout << "real: " << matToShow.imag.getMat(cv::ACCESS_READ).at<float>(0,0) <<std::endl;
    std::cout << "imag: " << matToShow.real.getMat(cv::ACCESS_READ).at<float>(0,0) <<std::endl;

    /* The goal is to display 4 images:
     *   |--------+--------|
     *   |  real  +  imag  |
     *   |--------+--------|
     *   |   abs  +  phase |
     *   |--------+--------|
     */

    // Compute phase and amplitude
    cvc::cMat phaseMatC = cvc::angle(matToShow);
    cvc::cMat ampMatC = cvc::abs(matToShow);

    // Normalize real and imaginary parts for display
    cv::normalize(matToShow.real, realMat, 0, 255, CV_MINMAX);
    cv::normalize(matToShow.imag, imagMat, 0, 255, CV_MINMAX);
    cv::normalize(ampMatC.real, ampMat, 0, 255, CV_MINMAX);
    cv::normalize(phaseMatC.real, phaseMat, 0, 255, CV_MINMAX);

    // Convert to 8-bit for display
    realMat.convertTo(realMat, CV_8U);
    imagMat.convertTo(imagMat, CV_8U);
    ampMat.convertTo(ampMat, CV_8U);
    phaseMat.convertTo(phaseMat, CV_8U);

    int w = 2; // # cols in subplots
    int h = 2; // # rows in subplots
    double paddingFactor = 0.1;

    int rows = cv::min(matToShow.real.rows,400);
    int cols = cv::min(matToShow.real.cols,400);

    int rowsPadded = std::round(rows*(1.0+(2 * paddingFactor)));
    int colsPadded = std::round(cols*(1.0+(2 * paddingFactor)));

    cv::Mat subPlotImage = cv::Mat(h * rowsPadded, w * colsPadded, CV_8U);
    subPlotImage.setTo(0);

    cv::Rect r11 = cv::Rect(std::round(paddingFactor * cols), std::round(paddingFactor * rows), cols, rows);
    cv::Rect r12 = cv::Rect(std::round(paddingFactor * cols + colsPadded), std::round(paddingFactor * rows), cols, rows);
    cv::Rect r21 = cv::Rect(std::round(paddingFactor * cols), std::round(paddingFactor * rows + rowsPadded), cols, rows);
    cv::Rect r22 = cv::Rect(std::round(paddingFactor * cols + colsPadded), std::round(paddingFactor * rows + rowsPadded), cols, rows);

    std::cout << r11 << std::endl;
    std::cout << r12 << std::endl;
    std::cout << r21 << std::endl;
    std::cout << r22 << std::endl;

    realMat.copyTo(subPlotImage(r11));
    imagMat.copyTo(subPlotImage(r12));
    ampMat.copyTo(subPlotImage(r21));
    phaseMat.copyTo(subPlotImage(r22));

    // Window functions
    cv::startWindowThread();
    cv::namedWindow(windowTitle, cv::WINDOW_NORMAL);

    // Mouse callback
    cv::setMouseCallback(windowTitle, cvc::mouseCallback_cmshow, &subPlotImage);;

    // Show and wait for key to close
    cv::imshow(windowTitle, subPlotImage);
    cv::waitKey();
    cv::destroyAllWindows();
}

// Mouse callback for showImg
void cvc::mouseCallback_cmshow( int event, int x, int y, int, void* param)
{
	cv::Mat* imgPtr = (cv::Mat*) param;
	cv::Mat image;
	imgPtr->copyTo(image);
	image.convertTo(image,CV_64F); // to keep types consitant

	switch (event)
	{
		case ::CV_EVENT_LBUTTONDOWN:
		{
			// Pretty printing of complex matricies
			std::printf("x:%d y:%d: %.4f \n\n", x, y, image.at<double>(y,x));
			break;
		}
		case ::CV_EVENT_RBUTTONDOWN:
		{
			break;
		}
	default:
		return;
	}
}

cvc::cMat cvc::abs(cvc::cMat& inMat)
{
  cv::UMat tmp1;
  cv::UMat tmp2;
  cv::UMat tmp3;

  cv::multiply(inMat.real, inMat.real, tmp1);
  cv::multiply(inMat.imag, inMat.imag, tmp2);
  cv::add(tmp1, tmp2, tmp3);

  inMat.imag = cv::UMat::zeros(inMat.size(),inMat.type());
  cv::sqrt(tmp3,inMat.real);

  return inMat;
}

cvc::cMat cvc::angle(cvc::cMat& inMat)
{
  cv::phase(inMat.real, inMat.imag, inMat.real);
  inMat.imag = cv::UMat::zeros(inMat.size(),inMat.type());
  return inMat;
}

cvc::cMat cvc::conj(cvc::cMat& inMat)
{
  cv::multiply(-1.0, inMat.imag, inMat.imag);
  return inMat;
}
