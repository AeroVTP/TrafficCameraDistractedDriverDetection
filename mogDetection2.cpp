/*
 * mogDetection2.cpp
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
#include "slidingWindowNeighborDetector.h"

#include "mogDetection.h"
#include "mogDetection2.h"

#include "cannyContourDetector.h"

//namespaces for convenience
using namespace cv;
using namespace std;

extern int i;

//defining format of data sent to threads
struct thread_data {
	//include int for data passing
	int data;
};

//method to do background subtraction with MOG 2
void *computeBgMog2(void *threadarg) {

	extern vector <Mat> globalFrames;
	extern int i;
	extern Ptr<BackgroundSubtractorMOG2> pMOG2Shadow;
	extern Mat mogDetection2GlobalFrame;
	extern int mogDetection2GlobalFrameCompletion;

	struct thread_data *data;
	data = (struct thread_data *) threadarg;

	//instantiating Mat objects
	Mat fgmaskShadow;
	Mat frameToResizeShadow;

	//copying into tmp variable
	globalFrames[i].copyTo(frameToResizeShadow);

	//performing background subtraction
	pMOG2Shadow->operator()(frameToResizeShadow, fgmaskShadow, .01);

	//performing sWND
	displayFrame("fgmaskShadow", fgmaskShadow);
	Mat fgmaskShadowSWND = slidingWindowNeighborDetector(fgmaskShadow,
			fgmaskShadow.rows / 10, fgmaskShadow.cols / 20);
	displayFrame("fgmaskShadowSWND", fgmaskShadowSWND);

	fgmaskShadowSWND = slidingWindowNeighborDetector(fgmaskShadowSWND,
			fgmaskShadowSWND.rows / 20, fgmaskShadowSWND.cols / 40);
	displayFrame("fgmaskShadowSWND2", fgmaskShadowSWND);

	//performing canny
	Mat fgMaskShadowSWNDCanny = cannyContourDetector(fgmaskShadowSWND);
	displayFrame("fgMaskShadowSWNDCanny2", fgMaskShadowSWNDCanny);

	//return canny
	mogDetection2GlobalFrame = fgMaskShadowSWNDCanny;

	//signal completion
	mogDetection2GlobalFrameCompletion = 1;
}


void mogDetection2ThreadHandler(bool buffer) {
	//instantiating multithread object
	pthread_t mogDetection2Thread;

	//instantiating multithread Data object
	struct thread_data threadData;

	//saving data into data object
	threadData.data = i;

	//creating threads
	int mogDetection2ThreadRC = pthread_create(&mogDetection2Thread, NULL,
			computeBgMog2, (void *) &threadData);
}


