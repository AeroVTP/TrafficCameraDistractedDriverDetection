/*
 * vibeBackgroundSubtraction.cpp
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
#include "cannyContourDetector.h"

//namespaces for convenience
using namespace cv;
using namespace std;

extern int i;
extern int bufferMemory;
extern vector <Mat> globalFrames;
extern Mat resizedFrame;
extern Mat vibeDetectionGlobalFrame;
extern int vibeDetectionGlobalFrameCompletion;
extern bgfg_vibe bgfg;
extern Mat vibeBckFrame;

const int trainingScalarFactor = 7;

//defining format of data sent to threads
struct thread_data {
	//include int for data passing
	int data;
};

//method to perform vibe background subtraction
void *computeVibeBackgroundThread(void *threadarg) {
 	struct thread_data *data;
	data = (struct thread_data *) threadarg;

	//instantiating Mat frame object
	Mat sWNDVibeCanny;

	//if done buffering
	if (i == bufferMemory) {
		//instantiating Mat frame object
		Mat resizedFrame;

		//saving current frame
		globalFrames[i].copyTo(resizedFrame);

		//initializing model
		bgfg.init_model(resizedFrame);

		//return tmp frame
		vibeDetectionGlobalFrame = sWNDVibeCanny;

		vibeDetectionGlobalFrameCompletion = 1;
	}

	else {
		//instantiating Mat frame object
		Mat resizedFrame;

		//saving current frame
		globalFrames[i].copyTo(resizedFrame);

		//processing model
		vibeBckFrame = *bgfg.fg(resizedFrame);

		displayFrame("vibeBckFrame", vibeBckFrame);

		//performing sWND
		Mat sWNDVibe = slidingWindowNeighborDetector(vibeBckFrame,
				vibeBckFrame.rows / 10, vibeBckFrame.cols / 20);
		displayFrame("sWNDVibe1", sWNDVibe);

		//performing sWND
		sWNDVibe = slidingWindowNeighborDetector(vibeBckFrame,
				vibeBckFrame.rows / 20, vibeBckFrame.cols / 40);
		displayFrame("sWNDVibe2", sWNDVibe);

		Mat sWNDVibeCanny = sWNDVibe;

		if (i > bufferMemory * trainingScalarFactor - 1) {
			//performing canny
			Mat sWNDVibeCanny = cannyContourDetector(sWNDVibe);
			displayFrame("sWNDVibeCannycanny2", sWNDVibeCanny);
		}

		//saving processed frame
		vibeDetectionGlobalFrame = sWNDVibeCanny;

		//signalling completion
		vibeDetectionGlobalFrameCompletion = 1;
	}
}

void vibeBackgroundSubtractionThreadHandler(bool buffer) {
	//instantiating multithread object
	pthread_t vibeBackgroundSubtractionThread;

	//instantiating multithread Data object
	struct thread_data threadData;

	//saving data into data object
	threadData.data = i;

	//creating threads
	int vibeBackgroundThreadRC = pthread_create(
			&vibeBackgroundSubtractionThread, NULL, computeVibeBackgroundThread,
			(void *) &threadData);
}

