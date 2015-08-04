/*
 * grayScaleFrameMedian.cpp
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

//defining format of data sent to threads
struct thread_data {
	//include int for data passing
	int data;
};

//thread to calculate median of image
void *calcMedianImage(void *threadarg) {
	extern vector <Mat> globalGrayFrames;
	extern int i;
	extern Mat backgroundFrameMedian;
	extern Mat finalTrackingFrame;
	extern Mat drawAnomalyCar;
	extern Mat backgroundFrameMedianColor;
	extern int bufferMemory;
	extern int medianImageCompletion;

	//defining data structure to read in info to new thread
	struct thread_data *data;
	data = (struct thread_data *) threadarg;

	//performing deep copy
	globalGrayFrames[i].copyTo(backgroundFrameMedian);

	//variables to display completion
	double displayPercentageCounter = 0;
	double activeCounter = 0;

	//calculating number of runs
	for (int j = 0; j < backgroundFrameMedian.rows; j++) {
		for (int a = 0; a < backgroundFrameMedian.cols; a++) {
			for (int t = (i - bufferMemory); t < i; t++) {
				displayPercentageCounter++;
			}
		}
	}

	//stepping through all pixels
	for (int j = 0; j < backgroundFrameMedian.rows; j++) {
		for (int a = 0; a < backgroundFrameMedian.cols; a++) {
			//saving all pixel values
			vector<int> pixelHistory;

			//moving through all frames stored in buffer
			for (int t = (i - bufferMemory); t < i; t++) {
				//Mat to store current frame to process
				Mat currentFrameForMedianBackground;

				//copy current frame
				globalGrayFrames.at(i - t).copyTo(
						currentFrameForMedianBackground);

				//save pixel into pixel history
				pixelHistory.push_back(
						currentFrameForMedianBackground.at<uchar>(j, a));

				//increment for load calculations
				activeCounter++;
			}

			//calculate median value and store in background image
			backgroundFrameMedian.at<uchar>(j, a) = calcMedian(pixelHistory);
		}

		//display percentage completed
		cout << ((activeCounter / displayPercentageCounter) * 100)
				<< "% Median Image Scanned" << endl;

	}

	//saving background to write on
	backgroundFrameMedian.copyTo(finalTrackingFrame);
	backgroundFrameMedian.copyTo(drawAnomalyCar);
	backgroundFrameMedian.copyTo(backgroundFrameMedianColor);

	//signal thread completion
	medianImageCompletion = 1;
}


//method to perform median on grayscale images
void grayScaleFrameMedian() {
	extern bool debug;
	extern int i;
	extern Mat backgroundFrameMedian;

	if (debug)
		cout << "Entered gray scale median" << endl;

	//instantiating multithread object
	pthread_t medianImageThread;

	//instantiating multithread Data object
	struct thread_data threadData;

	//saving data into multithread
	threadData.data = i;

	//creating thread to calculate median of image
	pthread_create(&medianImageThread, NULL, calcMedianImage,
			(void *) &threadData);

	//save median image
	imwrite((currentDateTime() + "medianBackgroundImage.jpg"),
			backgroundFrameMedian);
}



