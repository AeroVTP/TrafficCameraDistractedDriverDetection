/*
 * thresholdFrame.cpp
 *
 *  Created on: Aug 4, 2015
 *      Author: Vidur
 */

//include opencv library files
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/gpu/gpu.hpp"
#include <opencv2/nonfree/ocl.hpp>
#include <opencv/cv.h>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/core/core.hpp>
#include "bgfg_vibe.hpp"

//include c++ files
#include <iostream>
#include <fstream>
#include <ctime>
#include <time.h>
#include <thread>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <limits>
#include <math.h>
#include <algorithm>
#include <vector>
#include <pthread.h>
#include <cstdlib>

#include "averageCoordinates.h"
#include "checkLanePosition.h"
#include "analyzeMovement.h"
#include "displayFrame.h"
#include "welcome.h"
#include "displayCoordinate.h"
#include "trackingML.h"
#include "displayCoordinates.h"
#include "processExit.h"
#include "computeRunTime.h"
#include "objectDetection.h"

#include "slidingWindowNeighborDetector.h"
#include "cannyContourDetector.h"
#include "opticalFlowFarneback.h"
#include "opticalFlowAnalysisObjectDetection.h"

#include "currentDateTime.h"

//namespaces for convenience
using namespace cv;
using namespace std;


//method to threshold standard frame
Mat thresholdFrame(Mat sourceDiffFrame, const int threshold) {
	//Mat to hold frame
	Mat thresholdFrame;

	//perform deep copy into destination Mat
	sourceDiffFrame.copyTo(thresholdFrame);

	//steping through pixels
	for (int j = 0; j < sourceDiffFrame.rows; j++) {
		for (int a = 0; a < sourceDiffFrame.cols; a++) {
			//if pixel value greater than threshold
			if (sourceDiffFrame.at<uchar>(j, a) > threshold) {
				//write to binary image
				thresholdFrame.at<uchar>(j, a) = 255;
			} else {
				//write to binary image
				thresholdFrame.at<uchar>(j, a) = 0;
			}
		}
	}

	//return thresholded frame
	return thresholdFrame;
}


