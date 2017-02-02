/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2014 Case Western Reserve University
 *    Russell C Jackson <rcj33@case.edu>
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Case Western Reserve Univeristy, nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <ros/ros.h>
#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

#include "vesselness_image_filter_gpu/vesselness_image_filter_kernels.h"
#include "vesselness_image_filter_gpu/vesselness_image_filter_gpu.h"
#include <vesselness_image_filter_common/vesselness_image_filter_common.h>


// This file defines the kernel functions used by the thin segmentation cuda code.
// @TODO move the kernel functions into their own file for conciseness.

// This device side function computes the pdf of a 2d gaussian 
// with a scaled identity covariance matrix.
__device__ float gaussFncGPU(float var, float x, float y)
{
  float result(expf(-x*x/(2.0f*var)-y*y/(2.0f*var))); 
  result /= (3.1415f*2.0f*var);
  return result;
}


// compute the XX gaussian Hessian kernel.
__global__ void genGaussHessKernel_XX(cv::cuda::PtrStepSzf output, float var, int offset)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int ixD = x-offset;
  int iyD = y-offset;
  if (x < output.cols && y < output.rows)
  {
    float gaussV = gaussFncGPU(var, ixD, iyD);
    float v = (ixD*ixD)/(var*var)*gaussV-1/(var)*gaussV;
    output(y, x) = v;
  }
}

// compute the XY gaussian Hessian kernel.
__global__ void genGaussHessKernel_XY(cv::cuda::PtrStepSzf output, float var, int offset)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int ixD = offset-x;
  int iyD = offset-y;
  if (x < output.cols && y < output.rows)
  {
    float gaussV = gaussFncGPU(var, ixD, iyD);
    float v = (iyD*ixD)/(var*var)*gaussV;
    output(y, x) = v;
  }
}

// compute the YY gaussian Hessian kernel.
__global__ void genGaussHessKernel_YY(cv::cuda::PtrStepSzf output, float var, int offset)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int ixD = x-offset;
  int iyD = y-offset;
  if (x < output.cols && y < output.rows)
  {
    float gaussV = gaussFncGPU(var, ixD, iyD);
    float v = (iyD*iyD)/(var*var)*gaussV-1/(var)*gaussV;
    output(y, x) = v;
  }
}

// compute eigen values.
__global__ void generateEigenValues(
  const cv::cuda::PtrStepSzf XX, const cv::cuda::PtrStepSzf XY, const cv::cuda::PtrStepSzf YY,
  cv::cuda::PtrStepSz<float2> output, float betaParam, float cParam)
{
  // get the image index.
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  if (x < output.cols && y < output.rows)
  {
    float V_mag(0.0);
    float aOut(0.0);
    float eig0(0.0);
    float eig1(0.0);
    float det(XX(y, x)*YY(y, x)-XY(y, x)*XY(y, x));
    float b(-XX(y, x)-YY(y, x));
    float descriminant(sqrt(b*b-4*det));
    float r_Beta;
    float v_y(0.0);
    float v_x(1.0);
    if (descriminant > 0.000000001)
    {
      eig0 = (-b+descriminant)/(2);
      eig1 = (-b-descriminant)/(2);
      r_Beta = eig0/eig1;
      // find the dominant eigenvector:
      if (abs(r_Beta) > 1.0)
      {
        // indicates that eig0 is larger.
        r_Beta = 1/r_Beta;
        v_y = (eig0-XX(y, x))*v_x/(XY(y, x));
      }
      else v_y = (eig1-XX(y, x))*v_x/(XY(y, x));

      float a = atan2(v_y, v_x);
      if (a > 0.00)
      {
        aOut = (a);
      }
      else
      {
        aOut = (a+3.1415);
      }
    }
    else
    {
      eig0 = eig1 = -b/2;
      r_Beta = 1.0;
      v_x = 0.00;
      v_y = 1.0;
      aOut = 0.0;
    }

    // compute the output magnitude:
    V_mag = exp(-r_Beta*r_Beta/(betaParam))*(1-exp(-(eig0*eig0+eig1*eig1)/(cParam)));

    // assign the final output.
    output(y, x).x = aOut;
    output(y, x).y = V_mag;
  }
}


// Gaussian blurring function
__global__ void gaussAngBlur(
  const cv::cuda::PtrStepSz<float2> srcMat, cv::cuda::PtrStepSz<float2> dstMat,
  cv::cuda::PtrStepSzf gMat, int gaussOff)
{
  // get the image index.
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < srcMat.cols && y < srcMat.rows)
  {
    float val = 0.0;
    float2 dirPt;
    dirPt.x = 0.0;
    dirPt.y = 0.0;

    int gaussPixCount = (gaussOff*2+1);

    for (int gx = 0; gx < gMat.cols; gx++)
      for (int gy = 0; gy < gMat.rows; gy++)
      {
        int srcXPos = x-gaussOff+gx;
        int srcYPos = y-gaussOff+gy;

        // constant corner assumption:
        if (srcXPos < 0) srcXPos = 0;
        if (srcYPos < 0) srcYPos = 0;

        if (srcXPos >= srcMat.cols) srcXPos = srcMat.cols-1;
        if (srcYPos >= srcMat.rows) srcYPos = srcMat.rows-1;

        float tmpVal = srcMat(srcYPos, srcXPos).y*gMat(gy, gx);
        val += tmpVal;

        float tmpAngle = srcMat(srcYPos, srcXPos).x;

        float2 newDir;
        newDir.x =  tmpVal*cos(tmpAngle);
        newDir.y =  tmpVal*sin(tmpAngle);

        float tempNorm = sqrt(dirPt.x*dirPt.x + dirPt.y*dirPt.y);

        // find the cos between the two vectors;
        // This is used for scaling the sum.
        float dotResult = (newDir.x*dirPt.x+newDir.y*dirPt.y)/(tempNorm*tmpVal);

        if (dotResult < 0.0)  // -0.707...?
        {
            dirPt.x -= newDir.x;
            dirPt.y -= newDir.y;
        }
        else
        {
            dirPt.x += newDir.x;
            dirPt.y += newDir.y;
        }
      }
      dstMat(y, x).y = val;
      float newAngle = atan2(dirPt.y, dirPt.x);
      if (newAngle < 0.0) dstMat(y, x).x = (newAngle+3.1415);
      else dstMat(y, x).x = (newAngle);
  }
  return;
}



VesselnessNodeGPU::VesselnessNodeGPU(const char* subscriptionChar, const char* publicationChar):
  VesselnessNodeBase(subscriptionChar, publicationChar)
{
  outputChannels = 2;
  initKernels();
  setParamServer();
}

void VesselnessNodeGPU::initKernels()
{
  // reallocate the GpuMats
  cv::cuda::GpuMat tempGPU_XX;
  cv::cuda::GpuMat tempGPU_XY;
  cv::cuda::GpuMat tempGPU_YY;


  // reallocate the gpu matrices.
  tempGPU_XX.create(filterParameters.hessProcess.side, filterParameters.hessProcess.side, CV_32FC1);
  tempGPU_XY.create(filterParameters.hessProcess.side, filterParameters.hessProcess.side, CV_32FC1);
  tempGPU_YY.create(filterParameters.hessProcess.side, filterParameters.hessProcess.side, CV_32FC1);

  // initialize the hessian kernels:
  // @TODO use static_casting.
  int offset =  (int) floor((float)filterParameters.hessProcess.side/2);
  ROS_INFO("Assigning");
  dim3 kBlock(1, 1, 1);
  dim3 kThread(filterParameters.hessProcess.side, filterParameters.hessProcess.side, 1);
  genGaussHessKernel_XX<<<kBlock, kThread>>>(tempGPU_XX, filterParameters.hessProcess.variance, offset);
  genGaussHessKernel_XY<<<kBlock, kThread>>>(tempGPU_XY, filterParameters.hessProcess.variance, offset);
  genGaussHessKernel_YY<<<kBlock, kThread>>>(tempGPU_YY, filterParameters.hessProcess.variance, offset);

  // download the hessian kernels to the Host.
  tempGPU_XX.download(tempCPU_XX);
  tempGPU_XY.download(tempCPU_XY);
  tempGPU_YY.download(tempCPU_YY);

  // release the gpu hessian kernels.
  tempGPU_XX.release();
  tempGPU_XY.release();
  tempGPU_YY.release();

  // initialize and upload the filterParameters.postProcess Kernel:
  cv::Mat gaussKernel = cv::getGaussianKernel(
    filterParameters.postProcess.side, filterParameters.postProcess.variance, CV_32FC1);
  cv::Mat gaussOuter  = gaussKernel*gaussKernel.t();

  gaussG.upload(gaussOuter);

  // Finished...
  ROS_INFO("Allocated the GPU post processing kernels.");
  this->kernelReady = true;
}


// This function allocates the GPU mem to save time
cv::Size VesselnessNodeGPU::allocateMem(const cv::Size& size_)
{
  deallocateMem();
  imgAllocSize = size_;

  // allocate the other matrices.
  preOutput.create(size_, CV_32FC2);
  outputG.create(size_, CV_32FC2);
  inputG.create(size_, CV_8UC3);
  inputGreyG.create(size_, CV_8UC1);
  inputFloat255G.create(size_, CV_32FC1);
  inputFloat1G.create(size_, CV_32FC1);


  ones.create(size_, CV_32FC1);
  ones.setTo(cv::Scalar(255.0));

  // Allocate the page lock memory
  dstMatMem.create(size_, CV_32FC2);
  ROS_INFO("Allocated the memory");
  return size_;  
}

// This function allocates the GPU mem to save time
void VesselnessNodeGPU::deallocateMem()
{
  // input data
  inputG.release();
  inputGreyG.release();
  inputFloat255G.release();
  inputFloat1G.release();
  ones.release();

  // output data
  preOutput.release();
  outputG.release();

  dstMatMem.release();
}


void VesselnessNodeGPU::segmentImage(const cv::Mat &srcMat, cv::Mat &dstMat)
{
  if (kernelReady != true)
  {
    ROS_INFO("Unable to process image");
    return;
  }

  // compute the size of the image
  int iX, iY;

  iX = srcMat.cols;
  iY = srcMat.rows;

  // create a stream for processing.
  cv::cuda::Stream streamInfo;
  cudaStream_t cudaStream;

  // upload &  convert image to gray scale with a max of 1.0;
  inputG.upload(srcMat, streamInfo);
  cv::cuda::cvtColor(inputG, inputGreyG, CV_BGR2GRAY, 0, streamInfo);
  inputGreyG.convertTo(inputFloat255G, CV_32FC1, 1.0, 0.0, streamInfo);
  cv::cuda::divide(inputFloat255G, ones, inputFloat1G, 1.0, CV_32F, streamInfo);

  // compute the Hessian kernel filters.
  cv::cuda::createLinearFilter(CV_32F, CV_32F, tempCPU_XX)->apply(inputFloat1G, cXX, streamInfo);
  cv::cuda::createLinearFilter(CV_32F, CV_32F, tempCPU_YY)->apply(inputFloat1G, cYY, streamInfo);
  cv::cuda::createLinearFilter(CV_32F, CV_32F, tempCPU_XY)->apply(inputFloat1G, cXY, streamInfo);

  // @TODO implement the static_cast.
  int blockX = (int) ceil((double) iX /(16.0f));
  int blockY = (int) ceil((double) iY /(16.0f));

  dim3 eigBlock(blockX, blockY, 1);
  dim3 eigThread(16, 16, 1);

  // get the cuda stream access
  cudaStream = cv::cuda::StreamAccessor::getStream(streamInfo);


  generateEigenValues<<<eigBlock, eigThread, 0, cudaStream>>>
    (cXX, cXY, cYY, preOutput, filterParameters.betaParam, filterParameters.cParam);

  // Blur the result:
  // @TODO implement static_casting.
  int gaussOff = (int) floor(((float) filterParameters.postProcess.side)/2.0f);
  gaussAngBlur<<<eigBlock, eigThread, 0, cudaStream>>>(preOutput, outputG, gaussG, gaussOff);


  outputG.download(dstMatMem, streamInfo);
  streamInfo.waitForCompletion();

  // export the final result.
  cv::Mat tempDst;
  tempDst = dstMatMem.createMatHeader();
  dstMat = tempDst.clone();
}


// Destructor Function
VesselnessNodeGPU::~VesselnessNodeGPU()
{
  deallocateMem();
}
