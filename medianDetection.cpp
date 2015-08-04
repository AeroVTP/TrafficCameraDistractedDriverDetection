/*
 * medianDetection.cpp
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

#include "generateBackgroundImage.h"

#include "imageSubtraction.h"

//namespaces for convenience
using namespace cv;
using namespace std;

//defining format of data sent to threads
struct thread_data {
	//include int for data passing
	int data;
};

//method to handle median image subtraction
Mat medianImageSubtraction(int FRAME_RATE) {
	//generate or read background image
	generateBackgroundImage(FRAME_RATE);

	//calculate image difference and return
	return imageSubtraction();
}

//method to handle median image subtraction
void *computeMedianDetection(void *threadarg) {
	extern Mat medianDetectionGlobalFrame;
	extern int FRAME_RATE;
	extern int medianDetectionGlobalFrameCompletion;

	struct thread_data *data;
	data = (struct thread_data *) threadarg;
	int tmp = data->data;

	medianDetectionGlobalFrame = medianImageSubtraction(FRAME_RATE);

	/*
	 //generate or read background image
	 generateBackgroundImage(FRAME_RATE);

	 //calculate image difference and save to global
	 medianDetectionGlobalFrame = imageSubtraction();
	 */

	medianDetectionGlobalFrameCompletion = 1;
}

void medianDetectionThreadHandler(int FRAME_RATE) {
	//instantiating multithread object
	pthread_t medianDetectionThread;

	//instantiating multithread Data object
	struct thread_data threadData;

	//saving data into data object
	threadData.data = FRAME_RATE;

	//creating threads
	int medianDetectionThreadRC = pthread_create(&medianDetectionThread, NULL,
			computeMedianDetection, (void *) &threadData);
}



