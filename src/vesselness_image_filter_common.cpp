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
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "vesselness_image_filter_common/vesselness_image_filter_common.h"


void VesselnessNodeBase::paramCallback(vesselness_image_filter::vesselness_params_Config &config, uint32_t level)
{
  ROS_INFO("Reconfigure request : %d %f %f %f %d %f",
    config.side_h,
    config.variance_h,
    config.beta,
    config.c,
    config.side_p,
    config.variance_p);

  // prepare the setting structures for use in the filter
  gaussParam hessParam_(config.variance_h, config.side_h);
  gaussParam postProcess_(config.variance_p, config.side_p);

  float betaParam_ = config.beta*config.beta;
  float cParam_    = config.c*config.c;

  // assign the new paramters and reinitializing the kernels.
  filterParameters =  segmentThinParam(hessParam_, postProcess_, betaParam_, cParam_);
  kernelReady = false;
  ROS_INFO("Reinitializing the kernels");
  this->initKernels();
  ROS_INFO("Updated and reinitialized the kernels");
}




VesselnessNodeBase::VesselnessNodeBase(const char* subscriptionChar, const char* publicationChar):
  it_(nh_),
  filterParameters(gaussParam(1.5, 5), gaussParam(2.0, 7), 0.1, 0.005),
  imgAllocSize(-1, -1),
  kernelReady(false),
  outputChannels(-1)
{
  // Subscribe to the input video feed.
  image_sub_ = it_.subscribe(subscriptionChar, 1,
      &VesselnessNodeBase::imgTopicCallback, this);

  // Publish to the output video feed.
  image_pub_ = it_.advertise(publicationChar, 1);
}


void  VesselnessNodeBase::imgTopicCallback(const sensor_msgs::ImageConstPtr& msg)
{
  cv_bridge::CvImagePtr cv_ptrIn;
  cv_bridge::CvImage   cv_Out;

  // Attempt to convert the image into an opencv form.
  // The image is assumed to be an 3-channel 24 bit pixel depth
  try
  {
      cv_ptrIn = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e)
  {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
  }

  // resize the images if necessary.
  if (cv_ptrIn->image.size().height != imgAllocSize.height || cv_ptrIn->image.size().width != imgAllocSize.width )
  {
    ROS_INFO("Resizing the allocated matrices");
    imgAllocSize = cv::Size(this->allocateMem(cv_ptrIn->image.size()));
  }

  // Actually segment the image.
  segmentImage(cv_ptrIn->image, outputImage);

  // The result is outputImage.
  // Publish this output
  // Fill in the headers and encoding type
  cv_Out.image = outputImage;
  // cv_Out.header =  cv_ptrIn->header;

  // only publish if the image type is reconized.
  bool publish(false);
  if (outputChannels == 1)
  {
    cv_Out.encoding = std::string("32FC1");
    publish = true;
  }
  else if (outputChannels == 2)
  {
    cv_Out.encoding = std::string("32FC2");
    publish = true;
  }
  else
  {
    ROS_INFO("The output is not properly set up");
  }

  // publish the outputdata now.
  if (publish)
  {
    image_pub_.publish(cv_Out.toImageMsg());
    ROS_INFO("published new image");
  }
}


// This is the blank destructor.
VesselnessNodeBase::~VesselnessNodeBase()
{
}


void VesselnessNodeBase::setParamServer()
{
    // initialize and connect to the parameter server.
    f = boost::bind(&VesselnessNodeBase::paramCallback, this, _1, _2);
    srv.setCallback(f);
}
