/*
 * gaussianMixtureModel.cpp
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

#include "slidingWindowNeighborDetector.h"
#include "cannyContourDetector.h"

using namespace std;
using namespace cv;

const int gmmScalarFactor = 7;

//defining format of data sent to threads
struct thread_data {
	//include int for data passing
	int data;
};

//method to calculate Gaussian image difference
void *calcGaussianMixtureModel(void *threadarg) {
	extern vector<Mat> globalFrames;
	extern int i;
	extern Mat gmmFrameRaw;
	extern Ptr<BackgroundSubtractorGMG> backgroundSubtractorGMM ;
	extern Mat binaryGMMFrame;
	extern Mat gmmTempSegmentFrame;
	extern Mat gmmFrame;
	extern int bufferMemory;
	extern Mat cannyGMM;
	extern int gaussianMixtureModelCompletion;

	//perform deep copy
	globalFrames[i].copyTo(gmmFrameRaw);

	//update model
	(*backgroundSubtractorGMM)(gmmFrameRaw, binaryGMMFrame);

	//save into tmp frame
	gmmFrameRaw.copyTo(gmmTempSegmentFrame);

	//add movement mask
	add(gmmFrameRaw, Scalar(0, 255, 0), gmmTempSegmentFrame, binaryGMMFrame);

	//save into display file
	gmmFrame = gmmTempSegmentFrame;

	//display frame
	displayFrame("GMM Frame", gmmFrame);

	//save mask as main gmmFrame
	gmmFrame = binaryGMMFrame;

	displayFrame("GMM Binary Frame", binaryGMMFrame);

	//if buffer built
	if (i > bufferMemory * 2) {
		//perform sWND
		gmmFrame = slidingWindowNeighborDetector(binaryGMMFrame,
				gmmFrame.rows / 5, gmmFrame.cols / 10);
		displayFrame("sWDNs GMM Frame 1", gmmFrame);

		gmmFrame = slidingWindowNeighborDetector(gmmFrame, gmmFrame.rows / 10,
				gmmFrame.cols / 20);
		displayFrame("sWDNs GMM Frame 2", gmmFrame);

		gmmFrame = slidingWindowNeighborDetector(gmmFrame, gmmFrame.rows / 20,
				gmmFrame.cols / 40);
		displayFrame("sWDNs GMM Frame 3", gmmFrame);

		Mat gmmFrameSWNDCanny = gmmFrame;

		if (i > bufferMemory * gmmScalarFactor - 1) {
			//perform Canny
			gmmFrameSWNDCanny = cannyContourDetector(gmmFrame);
			displayFrame("CannyGMM", gmmFrameSWNDCanny);
		}

		//save into canny
		cannyGMM = gmmFrameSWNDCanny;
	}

	//signal thread completion
	gaussianMixtureModelCompletion = 1;
}

//method to handle GMM thread
Mat gaussianMixtureModel() {
	extern int i;
	extern int bufferMemory;
	extern Mat cannyGMM;
	extern vector<Mat> globalFrames;

	//instantiate thread object
	pthread_t gaussianMixtureModelThread;

	//instantiating multithread Data object
	struct thread_data threadData;

	//save i data
	threadData.data = i;

	//create thread
	pthread_create(&gaussianMixtureModelThread, NULL, calcGaussianMixtureModel,
			(void *) &threadData);

	//return processed frame if completed
	if (i > bufferMemory * gmmScalarFactor)
		return cannyGMM;
	//return tmp frame if not finished
	else
		return globalFrames[i];
}



