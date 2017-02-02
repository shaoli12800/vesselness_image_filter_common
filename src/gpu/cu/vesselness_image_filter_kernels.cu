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
 *   * Neither the name of Case Western Reserve University, nor the names of its
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




#include "vesselness_image_filter_gpu/vesselness_image_filter_kernels.h"


// This file defines the kernel functions used by the thin segmentation cuda code.
// currently this file is UNUSED.
// @TODO use this file to define cuda kernel functions code readability.


// Need to remove this #define, replace it with a __local__ cuda function.
// #define gaussFncGPU(var,x,y) 1.0f/(3.1415f*2.0f*var)*((float) ));

/*
__device__ float gaussFncGPU(float var, float x, float y)
{
  float result(expf(-x*x/(2.0f*var)-y*y/(2.0f*var))); 
  result /= (3.1415f*2.0f*var);
  return result;
}

__global__ void genGaussHessKernel_XX(PtrStepSzf output,float var,int offset)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int ixD = x-offset;
    int iyD = y-offset;
    if (x < output.cols && y < output.rows)
    {
        float gaussV = gaussFncGPU(var,ixD,iyD);
        float v = (ixD*ixD)/(var*var)*gaussV-1/(var)*gaussV;
        output(y, x) = v; 
    }
}

__global__ void genGaussHessKernel_XY(PtrStepSzf output,float var,int offset)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    float ixD =(float) (offset-x); //offset-x;
    float iyD =(float) offset-y; //offset-y; //
    if (x < output.cols && y < output.rows)
    {
        float gaussV = gaussFncGPU(var,ixD,iyD);
        float v = (iyD*ixD)/(var*var)*gaussV;
        output(y,x) = v; 
    }
}


__global__ void genGaussHessKernel_YY(PtrStepSzf output,float var,int offset)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int ixD = x-offset;
    int iyD = y-offset;
    if (x < output.cols && y < output.rows)
    {
        float gaussV = gaussFncGPU(var,ixD,iyD);
        float v = (iyD*iyD)/(var*var)*gaussV-1/(var)*gaussV;
        output(y,x) = v; 
    }
}

__global__ void generateEigenValues(const PtrStepSzf XX,const PtrStepSzf XY,const PtrStepSzf YY,PtrStepSz<float2> output,float betaParam,float cParam)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < output.cols && y < output.rows)
    {
        float V_mag = 0.0;
        float aOut = 0.0;
        float eig0 = 0.0;
        float eig1 = 0.0;
        float det = XX(y,x)*YY(y,x)-XY(y,x)*XY(y,x);
        float b = -XX(y,x)-YY(y,x);
        float descriminant = sqrt(b*b-4*det);
        float r_Beta;
        float v_y = 0.0;
        float v_x = 1.0;
        if(descriminant > 0.000000001)
        {
            eig0 = (-b+descriminant)/(2);
            eig1 = (-b-descriminant)/(2);
            r_Beta = eig0/eig1;
            //find the dominant eigenvector:
            if(abs(r_Beta) > 1.0){  //indicates that eig0 is larger.
                r_Beta = 1/r_Beta;
                v_y = (eig0-XX(y,x))*v_x/(XY(y,x));
            }
            else v_y = (eig1-XX(y,x))*v_x/(XY(y,x));

            float a = atan2(v_y,v_x);
            if(a > 0.00)
            {
                aOut = (a); ///3.1415;
            }
            else
            {
                aOut = (a+3.1415); ///3.1415;
            }
        }
        else
        {
            eig0 = eig1 = -b/2;
            r_Beta = 1.0;
            v_x = 0.00;
            v_y = 1.0;
            aOut =0.0;
        }
        V_mag = exp(-r_Beta*r_Beta/(betaParam))*(1-exp(-(eig0*eig0+eig1*eig1)/(cParam)));

        output(y,x).x = aOut;
        output(y,x).y = V_mag;
        

        //output(y,x).x = eig0;
        //output(y,x).y = eig1;
        //output(y,x).z = aOut/(3.1415);
    }
}


//Gaussian blurring function
__global__ void gaussAngBlur(const PtrStepSz<float2> srcMat,PtrStepSz<float2> dstMat,PtrStepSzf gMat,int gaussOff)
{

    int x = threadIdx.x + blockIdx.x * blockDim.x; 
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < srcMat.cols && y < srcMat.rows)
    {
        float val = 0.0;
        float2 dirPt;
        dirPt.x = 0.0;
        dirPt.y = 0.0;

        int gaussPixCount= (gaussOff*2+1);


        for(int gx = 0; gx < gMat.cols; gx++)
            for(int gy = 0; gy < gMat.rows; gy++)
            {
                int srcXPos =x-gaussOff+gx;
                int srcYPos =y-gaussOff+gy;

                //constant corner assumption:
        if(srcXPos < 0) srcXPos = 0;
        if(srcYPos < 0) srcYPos = 0;
      
        if(srcXPos >= srcMat.cols) srcXPos = srcMat.cols-1;
        if(srcYPos >= srcMat.rows) srcYPos = srcMat.rows-1;

        float tmpVal = srcMat(srcYPos,srcXPos).y*gMat(gy,gx);
        val += tmpVal;
      
        float tmpAngle = srcMat(srcYPos,srcXPos).x; 

        float2 newDir;
        newDir.x =  tmpVal*cos(tmpAngle);
        newDir.y =  tmpVal*sin(tmpAngle);

        float tempNorm = sqrt(dirPt.x*dirPt.x+dirPt.y*dirPt.y); 
      
        //find the cos between the two vectors;
        float dotResult = (newDir.x*dirPt.x+newDir.y*dirPt.y)/(tempNorm*tmpVal);

        if(dotResult < -0.707)
        {
          dirPt.x-=newDir.x;
          dirPt.y-=newDir.y;
        }
        else
        {
          dirPt.x+=newDir.x;
          dirPt.y+=newDir.y;
        }
      }
      dstMat(y,x).y = val;  //val;
      float newAngle = atan2(dirPt.y,dirPt.x);
      if(newAngle < 0.0) dstMat(y, x).x = (newAngle+3.1415);
      else dstMat(y, x).x = (newAngle);
  }
  return;
}

*/

