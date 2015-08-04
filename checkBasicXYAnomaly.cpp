/*
 * checkBasicXYAnomaly.cpp
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

//namespaces for convenience
using namespace cv;
using namespace std;

//boolean to determine if anomaly is detected
bool checkBasicXYAnomaly(int xMovement, int yMovement, Point carPoint, double currentAngle) {

	//thresholds to determine if anomaly
	const double maxThreshold = 5;
	const double minThreshold = -5;
	const int maxMovement = 30;

	const int angleThresholdMax = 10;
	const int angleThresholdMin = -10;
	const int angleThreshold = 20;

	//if above angle threshold
	if (currentAngle > angleThreshold)
	{
		//write coordinate
		displayCoordinate(carPoint);
		cout << " !!!!!!!!!!!ANOMALY DETECTED (ANGLE)!!!!!!!!!!!" << endl;
		//is anomalous
		return true;
 	}
	else
 	{
 		//is normal
 		return false;
 	}
}


