/*
 * cannyContourDetector.cpp
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
#include "fillCoordinates.h"

//namespaces for convenience
using namespace cv;
using namespace std;

//method to draw canny contours
Mat cannyContourDetector(Mat srcFrame) {
	extern int FRAME_WIDTH;
	extern int FRAME_HEIGHT;

	//threshold for non-car objects or noise
	const int thresholdNoiseSize = 200;
	const int misDetectLargeSize = 600;

	extern int xLimiter;
	extern int yLimiter;
	extern int xFarLimiter;

	//instantiating Mat and Canny objects
	Mat canny;
	Mat cannyFrame;
	vector<Vec4i> hierarchy;
	typedef vector<vector<Point> > TContours;
	TContours contours;

	//run canny edge detector
	Canny(srcFrame, cannyFrame, 300, 900, 3);
	findContours(cannyFrame, contours, hierarchy, CV_RETR_CCOMP,
			CV_CHAIN_APPROX_NONE);

	//creating blank frame to draw on
	Mat drawing = Mat::zeros(cannyFrame.size(), CV_8UC3);

	//moments for center of mass
	vector<Moments> mu(contours.size());
	for (int i = 0; i < contours.size(); i++) {
		mu[i] = moments(contours[i], false);
	}

	//get mass centers:
	vector<Point2f> mc(contours.size());
	for (int i = 0; i < contours.size(); i++) {
		mc[i] = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
	}

	//for each detected contour
	for (int v = 0; v < contours.size(); v++) {
		//if large enough to be object
		if (arcLength(contours[v], true) > thresholdNoiseSize
				&& arcLength(contours[v], true) < misDetectLargeSize) {
			if((mc[v].x > xLimiter && mc[v].x < FRAME_WIDTH - xFarLimiter) && (mc[v].y > yLimiter && mc[v].y < FRAME_HEIGHT - yLimiter))
			{
				//draw object and circle center point
				drawContours(drawing, contours, v, Scalar(254, 254, 0), 2, 8,
						hierarchy, 0, Point());
				circle(drawing, mc[v], 4, Scalar(254, 254, 0), -1, 8, 0);
			}
			fillCoordinates(mc, xLimiter, yLimiter);
		}
	}

	//return image with contours
	return drawing;
}


