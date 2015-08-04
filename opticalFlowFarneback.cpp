/*
 * opticalFlowFarneback.cpp
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
#include "opticalFlowAnalysisObjectDetection.h"

#include "blurFrame.h"

//namespaces for convenience
using namespace cv;
using namespace std;

extern vector <Mat> globalFrames;
extern int i;
extern Mat cflow;
extern bool debug;
extern Mat flow;
extern int opticalFlowAnalysisObjectDetectionThreadCompletion;
extern int opticalFlowThreadCompletion;
extern Mat thresholdFrameOFA;

//defining format of data sent to threads
struct thread_data {
	//include int for data passing
	int data;
};

//method to draw optical flow, only should be called during demos
void drawOpticalFlowMap(const Mat& flow, Mat& cflowmap, double,
		const Scalar& color) {
	extern int opticalFlowDensityDisplay;

	//iterating through each pixel and drawing vector
	for (int y = 0; y < cflowmap.rows; y += opticalFlowDensityDisplay) {
		for (int x = 0; x < cflowmap.cols; x += opticalFlowDensityDisplay) {
			const Point2f& fxy = flow.at<Point2f>(y, x);
			line(cflowmap, Point(x, y),
					Point(cvRound(x + fxy.x), cvRound(y + fxy.y)), color);
			circle(cflowmap, Point(x, y), 0, color, -1);
		}
	}
	//display optical flow map
	displayFrame("RFDOFA", cflowmap);
 }

//method to perform optical flow analysis
void *computeOpticalFlowAnalysisThread(void *threadarg) {

	//reading in data sent to thread into local variable
	struct thread_data *data;
	data = (struct thread_data *) threadarg;
	int temp = data->data;

	//defining local variables for FDOFA
	Mat prevFrame, currFrame;
	Mat gray, prevGray;

	//saving images for OFA
	prevFrame = globalFrames[i - 1];
	currFrame = globalFrames[i];

	//blurring frames
	displayFrame("Pre blur", currFrame);
	currFrame = blurFrame("gaussian", currFrame, 15);
	displayFrame("Post blur", currFrame);
	prevFrame = blurFrame("gaussian", prevFrame, 15);

	//converting to grayscale
	cvtColor(currFrame, gray, COLOR_BGR2GRAY);
	cvtColor(prevFrame, prevGray, COLOR_BGR2GRAY);

	//calculating optical flow
	calcOpticalFlowFarneback(prevGray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

	//converting to display format
	cvtColor(prevGray, cflow, COLOR_GRAY2BGR);

	//perform OFA threshold
	opticalFlowAnalysisObjectDetection(flow, cflow);

	//draw optical flow map
	if (debug) {
 		//drawing optical flow vectors
		drawOpticalFlowMap(flow, cflow, 1.5, Scalar(0, 0, 255));
	}

	//wait for completion
	while (opticalFlowAnalysisObjectDetectionThreadCompletion != 1) {
	}

	//wait for completion
	opticalFlowAnalysisObjectDetectionThreadCompletion = 0;

	//signal completion
	opticalFlowThreadCompletion = 1;
}

//method to handle OFA thread
Mat opticalFlowFarneback() {

 	//instantiate thread object
	pthread_t opticalFlowFarneback;

	//instantiating multithread Data object
	struct thread_data threadData;

	//saving data to pass
	threadData.data = i;

	//create OFA thread
	pthread_create(&opticalFlowFarneback, NULL,
			computeOpticalFlowAnalysisThread, (void *) &threadData);

	//waiting for finish
	while (opticalFlowThreadCompletion != 1) {
	}

	//resetting completion variable
	opticalFlowThreadCompletion = 0;

	//return OFA frame
	return thresholdFrameOFA;
}



