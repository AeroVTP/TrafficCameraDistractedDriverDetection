/*
 * processCoordinates.cpp
 *
 *  Created on: Aug 3, 2015
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
#include "displayCoordinates.h"
#include "drawCoordinates.h"

//method to handle coordinates
void processCoordinates() {

	extern String fileTime;
	extern Mat finalTrackingFrame;
	extern vector<Point> detectedCoordinates;
	extern int numberOfCars;

	const int averageThreshold = 85;

	//draw raw coordinates
	drawCoordinates(detectedCoordinates, "1st Pass");

	//write to file
	imwrite(fileTime + "finalTrackingFrame.TIFF", finalTrackingFrame);

	//average points using threshold
	detectedCoordinates = averageCoordinates(detectedCoordinates, averageThreshold);

	//count number of cars
	numberOfCars = detectedCoordinates.size();

	//draw processed coordinates
	drawCoordinates(detectedCoordinates, "2nd Pass");
}


