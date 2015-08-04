/*
 * currentDateTime.cpp
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

#include "gaussianMixtureModel.h"
#include "opticalFlowFarneback.h"

#include "vibeBackgroundSubtraction.h"

#include "mogDetection.h"
#include "mogDetection2.h"

#include "medianDetection.h"

#include "grayScaleFrameMedian.h"
#include "calcMedian.h"
#include "currentDateTime.h"

//namespaces for convenience
using namespace cv;
using namespace std;

//method that returns date and time as a string to tag txt files
const string currentDateTime() {
	extern string fileTime;

	//creating time object that reads current time
	time_t now = time(0);

	//creating time structure
	struct tm tstruct;

	//creating a character buffer of 80 characters
	char buf[80];

	//checking current local time
	tstruct = *localtime(&now);

	//writing time to string
	strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

	fileTime = buf;

	//returning the string with the time
	return buf;
}


