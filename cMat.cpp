#include "cMat.h"
#include <stdio.h>
#include <cmath>
#include <iostream>

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

    cv::Mat realMat = matToShow.real.getMat(cv::ACCESS_RW).clone();
    cv::Mat imagMat = matToShow.imag.getMat(cv::ACCESS_RW).clone();
    cv::Mat ampMat   = cvc::abs(matToShow).real.getMat(cv::ACCESS_RW).clone();
    cv::Mat phaseMat = cvc::angle(matToShow).real.getMat(cv::ACCESS_RW).clone();


    /* The goal is to display 4 images:
     *   |--------+--------|
     *   |  real  |  imag  |
     *   |--------+--------|
     *   |   abs  |  phase |
     *   |--------+--------|
     */

    // Normalize real and imaginary parts for display
    cv::normalize(realMat, realMat, 0.0, 255.0, CV_MINMAX);
    cv::normalize(imagMat, imagMat, 0.0, 255.0, CV_MINMAX);
    cv::normalize(ampMat, ampMat, 0.0, 255.0, CV_MINMAX);
    cv::normalize(phaseMat, phaseMat, 0.0, 255.0, CV_MINMAX);

    cv::UMat realMatDisp;
    cv::UMat imagMatDisp;
    cv::UMat ampMatDisp;
    cv::UMat phaseMatDisp;

    // Convert to 8-bit for display
    realMat.convertTo(realMatDisp, CV_8U);
    imagMat.convertTo(imagMatDisp, CV_8U);
    ampMat.convertTo(ampMatDisp, CV_8U);
    phaseMat.convertTo(phaseMatDisp, CV_8U);

    int w = 2; // # cols in subplots
    int h = 2; // # rows in subplots
    double paddingFactor = 0.1;

    int rows = cv::min(matToShow.real.rows, 400);
    int cols = cv::min(matToShow.real.cols, 400);

    int rowsPadded = std::round(1.0 + rows * (1.0 + (2 * paddingFactor)));
    int colsPadded = std::round(1.0 + cols * (1.0 + (2 * paddingFactor)));

    cv::Mat subPlotImage = cv::Mat(h * rowsPadded, w * colsPadded, CV_8U);
    cv::Mat subPlotVals = cv::Mat(h * rowsPadded, w * colsPadded, CV_8U);
    subPlotImage.setTo(0);
    subPlotVals.setTo(0);

    cv::Rect r11 = cv::Rect(std::round(paddingFactor * cols), std::round(paddingFactor * rows), cols, rows);
    cv::Rect r12 = cv::Rect(std::round(paddingFactor * cols + colsPadded), std::round(paddingFactor * rows), cols, rows);
    cv::Rect r21 = cv::Rect(std::round(paddingFactor * cols), std::round(paddingFactor * rows + rowsPadded), cols, rows);
    cv::Rect r22 = cv::Rect(std::round(paddingFactor * cols + colsPadded), std::round(paddingFactor * rows + rowsPadded), cols, rows);

    /*
    std::cout << r11 << std::endl;
    std::cout << r12 << std::endl;
    std::cout << r21 << std::endl;
    std::cout << r22 << std::endl;
    */

    realMatDisp.copyTo(subPlotImage(r11));
    imagMatDisp.copyTo(subPlotImage(r12));
    ampMatDisp.copyTo(subPlotImage(r21));
    phaseMatDisp.copyTo(subPlotImage(r22));

    realMat.copyTo(subPlotVals(r11));
    imagMat.copyTo(subPlotVals(r12));
    ampMat.copyTo(subPlotVals(r21));
    phaseMat.copyTo(subPlotVals(r22));

    // Window functions
    cv::startWindowThread();
    cv::namedWindow(windowTitle, cv::WINDOW_NORMAL);

    // Mouse callback
    cv::setMouseCallback(windowTitle, cvc::mouseCallback_cmshow, &subPlotVals);

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

cvc::cMat cvc::abs(const cvc::cMat& inMat)
{
  cvc::cMat output;
  cv::UMat tmp1;

  output.set_size(inMat.size());
  output.set_type(inMat.type());
  output.imag = cv::UMat::zeros(inMat.size(),inMat.type());
  cv::multiply(inMat.real, inMat.real, output.real);
  cv::multiply(inMat.imag, inMat.imag, tmp1);
  cv::add(tmp1, output.real, output.real);
  cv::sqrt(output.real,output.real);
  return output;
}

cvc::cMat cvc::angle(const cvc::cMat& inMat)
{
  cvc::cMat output;
  output.set_size(inMat.size());
  output.set_type(inMat.type());
  output.imag = cv::UMat::zeros(inMat.size(),inMat.type());
  cv::phase(inMat.real, inMat.imag, output.real);
  return output;
}

cvc::cMat cvc::conj(const cvc::cMat& inMat)
{
    cvc::cMat output;
    output.set_size(inMat.size());
    output.set_type(inMat.type());
    inMat.real.copyTo(output.real);
    cv::multiply(-1.0, inMat.imag, output.imag);
    return output;
}

cvc::cMat cvc::exp(const cvc::cMat& inMat)
{
    cvc::cMat output(inMat.real);

    for(int row = 0; row < inMat.real.rows; row++)
  	{
      const double* in_im_row = inMat.imag.getMat(cv::ACCESS_RW).ptr<double>(row);  // Input
      double* out_re_row = output.real.getMat(cv::ACCESS_RW).ptr<double>(row);   // Output real
      double* out_im_row = output.imag.getMat(cv::ACCESS_RW).ptr<double>(row);   // Output imag

      for(int col = 0; col < inMat.real.cols; col++)
      {
          std::complex<double> z = std::exp(std::complex<double> (out_re_row[col],in_im_row[col]));
          out_re_row[col] = z.real();
          out_im_row[col] = z.imag();
      }
  	}
    return output;
}

cvc::cMat cvc::log(const cvc::cMat& inMat)
{
    cvc::cMat output(inMat.real);

    for(int row = 0; row < inMat.real.rows; row++)
  	{
      const double* in_im_row = inMat.imag.getMat(cv::ACCESS_RW).ptr<double>(row);  // Input
      double* out_re_row = output.real.getMat(cv::ACCESS_RW).ptr<double>(row);   // Output real
      double* out_im_row = output.imag.getMat(cv::ACCESS_RW).ptr<double>(row);   // Output imag

      for(int col = 0; col < inMat.real.cols; col++)
      {
          std::complex<double> z = std::log(std::complex<double> (out_re_row[col],in_im_row[col]));
          out_re_row[col] = z.real();
          out_im_row[col] = z.imag();
      }
  	}
    return output;
}

cvc::cMat cvc::vec(const cvc::cMat& inMat)
{
    int16_t row_num = inMat.real.rows*inMat.real.cols;
    return *new cvc::cMat (inMat.real.reshape(1,row_num),inMat.imag.reshape(1,row_num));
}

cvc::cMat cvc::reshape(const cvc::cMat& inMat, const int rows)
{
    return *new cvc::cMat (inMat.real.reshape(1,rows),inMat.imag.reshape(1,rows));
}

/*
 * Performs a 1D FFT.
 *
 * @param real              real part of data. Should be a row vector.
 * @param imag              imaginary part of data. Should be a row vector.
 * @param data              cMat row vector.
 */
//cvc::cMat cvc::fft(cv::Mat real, cv::Mat imag) {

//}

/*
 * Performs a 2D FFT.
 */
 /*
cvc::cMat cvc::fft2(cvc::cMat& inMat) {

    // Perform a 1D FFT on each row, then perform a 1D FFT on each column of the
    // resulting matrix.

    //use library fft

    cvc::cMat result (inMat.getSize(), 5);
    for (int r = 0; r < result.getSize().height; r++) {
        cMat ft = cvc::fft(inMat.getRealRow(r), inMat.getImagRow(r));
        result.setRow(ft, r);
    }

    for (int c = 0; c < result.getSize().width; c++) {
        cMat ft = cvc::fft(inMat.getRealCol(c).t(), inMat.getImagCol(c).t());
        result.setCol(ft.t(), c);
    }

    return result;


}
*/

cvc::cMat cvc::fft2(cvc::cMat& inMat)
{
    cvc::cMat output(inMat.size(), inMat.type());

    cv::UMat biChannel = inMat.getBiChannel();
    cv::UMat biChannel_ft = biChannel.clone();
    cv::dft(biChannel, biChannel_ft, cv::DFT_COMPLEX_OUTPUT);
    output.setFromBiChannel(biChannel_ft);
    return output;
}

cvc::cMat cvc::ifft2(cvc::cMat& inMat)
{
    cvc::cMat output(inMat.size(), inMat.type());
    cv::UMat biChannel = inMat.getBiChannel();
    cv::UMat biChannel_ft = biChannel.clone();
    cv::dft(biChannel, biChannel_ft, cv::DFT_INVERSE | cv::DFT_COMPLEX_OUTPUT | cv::DFT_SCALE);
    output.setFromBiChannel(biChannel_ft);
    return output;
}


void cvc::fftshift(cvc::cMat& input, cvc::cMat& output)
{
	 	cvc::circularShift(input, output, std::floor((double) input.cols()/2), std::floor((double) input.rows()/2));
}

void cvc::ifftshift(cvc::cMat& input, cvc::cMat& output)
{
	 	cvc::circularShift(input, output, std::ceil((double) input.cols()/2), std::ceil((double) input.rows()/2));
}


void cvc::circularShift(cvc::cMat& input, cvc::cMat& output, int16_t x, int16_t y)
{
  if (output.real.empty())
    output = cvc::cMat(input.size(), input.type());

  cv::Mat input1_r = input.real.getMat(cv::ACCESS_READ);
  cv::Mat input1_i = input.imag.getMat(cv::ACCESS_READ);

  cv::Mat output1_r = output.real.getMat(cv::ACCESS_RW);
  cv::Mat output1_i = output.imag.getMat(cv::ACCESS_RW);

  int16_t w = input1_r.cols;
  int16_t h = input1_r.rows;

  int16_t shiftR = x % w;
  int16_t shiftD = y % h;

  if (shiftR < 0)//if want to shift in -x direction
      shiftR += w;

  if (shiftD < 0)//if want to shift in -y direction
      shiftD += h;

  cv::Rect gate1(0, 0, w-shiftR, h-shiftD);//rect(x, y, width, height)
  cv::Rect out1(shiftR, shiftD, w-shiftR, h-shiftD);

  cv::Rect gate2(w-shiftR, 0, shiftR, h-shiftD);
  cv::Rect out2(0, shiftD, shiftR, h-shiftD);

  cv::Rect gate3(0, h-shiftD, w-shiftR, shiftD);
  cv::Rect out3(shiftR, 0, w-shiftR, shiftD);

  cv::Rect gate4(w-shiftR, h-shiftD, shiftR, shiftD);
  cv::Rect out4(0, 0, shiftR, shiftD);

  // Generate pointers
  cv::Mat shift1_r, shift1_i, shift2_r, shift2_i, shift3_r, shift3_i, shift4_r, shift4_i;

  if (&input == &output) // Check if matricies have same pointer
  {
    shift1_r = input1_r(gate1).clone();
    shift1_i = input1_i(gate1).clone();

  	shift2_r = input1_r(gate2).clone();
    shift2_i = input1_i(gate2).clone();

  	shift3_r = input1_r(gate3).clone();
    shift3_i = input1_i(gate3).clone();

  	shift4_r = input1_r(gate4).clone();
    shift4_i = input1_i(gate4).clone();
  }
  else // safe to shallow copy
  {
    shift1_r = input1_r(gate1);
    shift1_i = input1_i(gate1);

    shift2_r = input1_r(gate2);
    shift2_i = input1_i(gate2);

	  shift3_r = input1_r(gate3);
    shift3_i = input1_i(gate3);

    shift4_r = input1_r(gate4);
    shift4_i = input1_i(gate4);
  }

  // Copy to result
	shift1_r.copyTo(cv::Mat(output1_r, out1));
    shift1_i.copyTo(cv::Mat(output1_i, out1));

	shift2_r.copyTo(cv::Mat(output1_r, out2));
    shift2_i.copyTo(cv::Mat(output1_i, out2));

	shift3_r.copyTo(cv::Mat(output1_r, out3));
    shift3_i.copyTo(cv::Mat(output1_i, out3));

	shift4_r.copyTo(cv::Mat(output1_r, out4));
    shift4_i.copyTo(cv::Mat(output1_i, out4));
}
