/*
 * opticalFlowAnalysisObjectDetection.cpp
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
#include "blurFrame.h"

#include "slidingWindowNeighborDetector.h"
#include "cannyContourDetector.h"

//namespaces for convenience
using namespace cv;
using namespace std;

extern vector<Mat> globalFrames;
extern vector<Mat> globalGrayFrames;
extern int i;
extern Mat flow;
extern Mat cflow;
extern bool debug;
extern int opticalFlowAnalysisObjectDetectionThreadCompletion;
extern int opticalFlowThreadCompletion;
extern Mat ofaGlobalHeatMap;
extern Mat thresholdFrameOFA;

//defining format of data sent to threads
struct thread_data {
	//include int for data passing
	int data;
};

//method to perform OFA threshold on Mat
void *computeOpticalFlowAnalysisObjectDetection(void *threadarg) {

	//reading in data sent to thread into local variable
	struct opticalFlowThreadData *data;
	data = (struct opticalFlowThreadData *) threadarg;

	Mat ofaObjectDetection;

	//deep copy grayscale frame
	globalGrayFrames.at(i - 1).copyTo(ofaObjectDetection);

	//set threshold
	const double threshold = 10000;

	//iterating through OFA pixels
	for (int j = 0; j < cflow.rows; j++) {
		for (int a = 0; a < cflow.cols; a++) {
			const Point2f& fxy = flow.at<Point2f>(j, a);

			//if movement is greater than threshold
			if ((sqrt((abs(fxy.x) * abs(fxy.y))) * 10000) > threshold) {
				//write to binary image
				ofaObjectDetection.at<uchar>(j, a) = 255;
			} else {
				//write to binary image
				ofaObjectDetection.at<uchar>(j, a) = 0;
			}
		}
	}

	//performing sWND
	displayFrame("OFAOBJ pre", ofaObjectDetection);

	ofaObjectDetection = slidingWindowNeighborDetector(ofaObjectDetection,
			ofaObjectDetection.rows / 10, ofaObjectDetection.cols / 20);
	displayFrame("sWNDFrame1", ofaObjectDetection);

	ofaObjectDetection = slidingWindowNeighborDetector(ofaObjectDetection,
			ofaObjectDetection.rows / 20, ofaObjectDetection.cols / 40);
	displayFrame("sWNDFrame2", ofaObjectDetection);

	ofaObjectDetection = slidingWindowNeighborDetector(ofaObjectDetection,
			ofaObjectDetection.rows / 30, ofaObjectDetection.cols / 60);
	displayFrame("sWNDFrame3", ofaObjectDetection);

	//saving into heat map
	ofaObjectDetection.copyTo(ofaGlobalHeatMap);

	//running canny detector
	thresholdFrameOFA = cannyContourDetector(ofaObjectDetection);
	displayFrame("sWNDFrameCanny", thresholdFrameOFA);

	//signal thread completion
	opticalFlowAnalysisObjectDetectionThreadCompletion = 1;
}

//method to handle OFA threshold on Mat thread
void opticalFlowAnalysisObjectDetection(Mat& cflowmap, Mat& flow) {
	//instantiating multithread object
	pthread_t opticalFlowAnalysisObjectDetectionThread;

	//instantiating multithread Data object
	struct thread_data threadData;

	//saving data to pass
	threadData.data = i;

	//creating optical flow object thread
	pthread_create(&opticalFlowAnalysisObjectDetectionThread, NULL,
			computeOpticalFlowAnalysisObjectDetection, (void *) &threadData);

}



