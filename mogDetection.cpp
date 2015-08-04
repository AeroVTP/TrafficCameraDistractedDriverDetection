/*
 * mogDetection.cpp
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

//namespaces for convenience
using namespace cv;
using namespace std;

//defining format of data sent to threads
struct thread_data {
	//include int for data passing
	int data;
};


//method to do background subtraction with MOG 1
void *computeBgMog1(void *threadarg) {

	extern BackgroundSubtractorMOG bckSubMOG;
	extern vector <Mat> globalFrames;
	extern int i;
	extern Mat mogDetection1GlobalFrame;
	extern int mogDetection1GlobalFrameCompletion;

	struct thread_data *data;
	data = (struct thread_data *) threadarg;

	//instantiating Mat objects
	Mat fgmask;
	Mat bck;
	Mat fgMaskSWNDCanny;

	//performing background subtraction
	bckSubMOG.operator()(globalFrames.at(i), fgmask, .01); //1.0 / 200);

	displayFrame("MOG Fg MAsk", fgmask);
	displayFrame("RCFrame", globalFrames[i]);

	//performing sWND
	Mat fgmaskSWND = slidingWindowNeighborDetector(fgmask, fgmask.rows / 10,
			fgmask.cols / 20);
	displayFrame("fgmaskSWND", fgmaskSWND);

	fgmaskSWND = slidingWindowNeighborDetector(fgmaskSWND, fgmaskSWND.rows / 20,
			fgmaskSWND.cols / 40);
	displayFrame("fgmaskSWNDSWND2", fgmaskSWND);

	fgmaskSWND = slidingWindowNeighborDetector(fgmaskSWND, fgmaskSWND.rows / 30,
			fgmaskSWND.cols / 60);
	displayFrame("fgmaskSWNDSWND3", fgmaskSWND);

	//performing canny
	fgMaskSWNDCanny = cannyContourDetector(fgmaskSWND);
	displayFrame("fgMaskSWNDCanny2", fgMaskSWNDCanny);

	//return canny
	mogDetection1GlobalFrame = fgMaskSWNDCanny;

	//signal completion
	mogDetection1GlobalFrameCompletion = 1;
}


void mogDetectionThreadHandler(bool buffer) {
	extern int i;

	//instantiating multithread object
	pthread_t mogDetectionThread;

	//instantiating multithread Data object
	struct thread_data threadData;

	//saving data into data object
	threadData.data = i;

	//creating threads
	int mogDetectionThreadRC = pthread_create(&mogDetectionThread, NULL,
			computeBgMog1, (void *) &threadData);
}



