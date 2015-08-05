/*
 * individualTracking.cpp
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
#include "processCoordinates.h"
#include "individualTracking.h"
#include "registerFirstCar.h"
#include "calculateDeviance.h"

//namespaces for convenience
using namespace cv;
using namespace std;

//method to handle individual tracking
void individualTracking() {
	extern int i;
	extern int bufferMemory;
	extern int mlBuffer;
	extern vector<vector<Point> > carCoordinates;
	extern vector<vector<Point> > vectorOfDetectedCars;
	extern vector<Point> detectedCoordinates;
	extern vector<vector<Point> > coordinateMemory;

	//distance threshold
 	const double distanceThreshold = 25;

 	//bool to show one car is registereds
	bool registerdOnce = false;

	//if ready to begin registering
	if (i == (bufferMemory + mlBuffer + 3) || ((carCoordinates.size() == 0) && i > (bufferMemory + mlBuffer)))
	{
 		registerFirstCar();
 	}

 	//if car is in scene
	else if (detectedCoordinates.size() > 0) {

  		//save into vector
		vectorOfDetectedCars.push_back(detectedCoordinates);
		coordinateMemory.push_back(detectedCoordinates);

		//calculate deviance of cars
		calculateDeviance();

		//analyze cars movement
		analyzeMovement();
	}
}


