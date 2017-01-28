#include "cMat.h"
#include <stdio.h>
#include <cmath>
#include <iostream>

inline cMat operator+(cMat lhs, const cMat& rhs) { lhs += rhs; return lhs; }
inline cMat operator+(cMat lhs, const double& rhs) { lhs += rhs; return lhs; }

inline cMat operator-(cMat lhs, const cMat& rhs) { lhs -= rhs; return lhs; }
inline cMat operator-(cMat lhs, const double& rhs) { lhs -= rhs; return lhs; }

inline cMat operator*(cMat lhs, const cMat& rhs) { lhs *= rhs; return lhs; }
inline cMat operator*(cMat lhs, const double& rhs) { lhs *= rhs; return lhs; }

inline cMat operator/(cMat lhs, const cMat& rhs) { lhs /= rhs; return lhs; }
inline cMat operator/(cMat lhs, const double& rhs) { lhs /= rhs; return lhs; }

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

void cmshow(cMat matToShow, std::string windowTitle)
{
    cv::Mat realMat;
    cv::Mat imagMat;

    // Normalize real and imaginary parts for display
    cv::normalize(matToShow.real.getMat(cv::ACCESS_READ), realMat, 0, 255, CV_MINMAX);
    cv::normalize(matToShow.imag.getMat(cv::ACCESS_READ), imagMat, 0, 255, CV_MINMAX);

    // Convert to 8-bit for display
    realMat.convertTo(realMat, CV_8U);
    imagMat.convertTo(imagMat, CV_8U);

    // Concatonate real and imaginary mat's
    cv::Mat displayMat;
    cv::hconcat(realMat, imagMat, displayMat);

    // Window functions
    cv::startWindowThread();
    cv::namedWindow(windowTitle, cv::WINDOW_NORMAL);

    // Mouse callback
    cv::setMouseCallback(windowTitle, mouseCallback_cmshow, &displayMat);;

    // Show and wait for key to close
    cv::imshow(windowTitle, displayMat);
    cv::waitKey();
    cv::destroyAllWindows();
}

// Mouse callback for showImg
void mouseCallback_cmshow( int event, int x, int y, int, void* param)
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



void showImg(cv::Mat m, std::string windowTitle, int16_t gv_cMap)
{

	cv::UMat displayMat;


}
