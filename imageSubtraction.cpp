/*
 * imageSubtraction.cpp
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
#include "slidingWindowNeighborDetector.h"
#include "cannyContourDetector.h"

#include "vibeBackgroundSubtraction.h"

#include "mogDetection.h"
#include "mogDetection2.h"

#include "medianDetection.h"

#include "grayScaleFrameMedian.h"
#include "calcMedian.h"
#include "currentDateTime.h"
#include "thresholdFrame.h"

//namespaces for convenience
using namespace cv;
using namespace std;

//method to perform simple image subtraction
Mat imageSubtraction() {

	extern vector<Mat> globalGrayFrames;
	extern int i;
	extern Mat backgroundFrameMedian;

	//subtract frames
	Mat tmpStore = globalGrayFrames[i] - backgroundFrameMedian;

	displayFrame("Raw imgSub", tmpStore);
	//threshold frames
	tmpStore = thresholdFrame(tmpStore, 50);
	displayFrame("Thresh imgSub", tmpStore);

	//perform sWND
	tmpStore = slidingWindowNeighborDetector(tmpStore, tmpStore.rows / 5,
			tmpStore.cols / 10);
	displayFrame("SWD", tmpStore);
	tmpStore = slidingWindowNeighborDetector(tmpStore, tmpStore.rows / 10,
			tmpStore.cols / 20);
	displayFrame("SWD2", tmpStore);
	tmpStore = slidingWindowNeighborDetector(tmpStore, tmpStore.rows / 20,
			tmpStore.cols / 40);
	displayFrame("SWD3", tmpStore);

	//perform canny
	tmpStore = cannyContourDetector(tmpStore);
	displayFrame("Canny Contour", tmpStore);

	//return frame
	return tmpStore;
}


