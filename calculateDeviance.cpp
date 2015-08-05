/*
 * calculateDeviance.cpp
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
#include "learnedCoordinate.h"

//method to calculate deviance
void calculateDeviance() {

	extern Mat backgroundFrameMedianColor;
	extern vector< vector<Point> > learnedCoordinates;

	//train LASM
	learnedCoordinate();

	//Mat to display LASM
	Mat tmpToDisplay;

	//save background
	backgroundFrameMedianColor.copyTo(tmpToDisplay);

	//cycle through LASM coordinates
	for (int v = 0; v < learnedCoordinates.size(); v++) {
		for (int j = 0; j < learnedCoordinates[v].size(); j++) {
			//read tmpPoint
			Point tmpPoint = learnedCoordinates[v][j];

			//create top left point
			Point tmpPointTopLeft(tmpPoint.x - .01, tmpPoint.y + .01);

			//create bottom right point
			Point tmpPointBottomRight(tmpPoint.x + .01, tmpPoint.y - .01);

			//create rectangle to display
	        rectangle( tmpToDisplay, tmpPointTopLeft, tmpPointBottomRight, Scalar(255, 255, 0), 1);
		}
	}
	displayFrame("Learned Path", tmpToDisplay, true);

}



