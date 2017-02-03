/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2016 Case Western Reserve University
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




#ifndef VESSELNESS_LIB_H
#define VESSELNESS_LIB_H


/**
 * @brief convert the segmented image into a displayable RGB format.
 *
 * This uses the CPU, (and is slower than a GPU version)
 *
 * @param src matrix 2 channel 32 bit float.
 * @param dst BGR displayable 8bit 3 channel unsigned int.
 */
void convertSegmentImageCPU(const cv::Mat&, cv::Mat&);

/**
 * @brief convert the single channel segmented image into a displayable format.
 *
 * This uses the CPU, (and is slower than a GPU version)
 *
 * @param src matrix 1 channel 32 bit float.
 * @param dst BGR displayable 8bit 3 channel unsigned int.
 */
void convertSegmentImageCPUBW(const cv::Mat&src, cv::Mat&dst);

/**
 * @brief Find the cutoff mean between the foreground and background of the segmented image.
 *
 * This uses OTSU's method on the CPU, (and is slower than a GPU version)
 *
 * @param src matrix 2 channel 32 bit float.
 * @param pointer to the cutoff mean.
 * @param number of iterations.
 */
void findOutputCutoff(const cv::Mat&, double *, int = 10);


/**
 * @brief This function computes the mean directiona nd magnitude for an image section.
 *
 * This uses the CPU, (and is slower than a GPU version)
 *
 * @param src matrix 2 channel 32 bit float.
 * @param Rectangle defining the region of interest.
 *
 * @return the mean vector
 */
cv::Point2f angleMagMean(const cv::Mat &, const cv::Rect &);

#endif  // VESSELNESS_LIB_H
