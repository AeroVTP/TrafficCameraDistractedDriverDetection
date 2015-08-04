/*
 * type2StrTest.cpp
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
#include "type2StrTest.h"

//namespaces for convenience
using namespace cv;
using namespace std;

//method to identify type of Mat based on identifier
string type2str(int type) {

	//string to return type of mat
	string r;

	//stats about frame
	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	//switch to determine Mat type
	switch (depth) {
		case CV_8U:
			r = "8U";
			break;
		case CV_8S:
			r = "8S";
			break;
		case CV_16U:
			r = "16U";
			break;
		case CV_16S:
			r = "16S";
			break;
		case CV_32S:
			r = "32S";
			break;
		case CV_32F:
			r = "32F";
			break;
		case CV_64F:
			r = "64F";
			break;
		default:
			r = "User";
			break;
	}

	//append formatting
	r += "C";
	r += (chans + '0');

	//return Mat type
	return r;
}


