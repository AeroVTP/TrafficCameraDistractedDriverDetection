/*
 * drawAllTracking.cpp
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

#include "displayFrame.h"

using namespace std;
using namespace cv;

//draw all coordinates
void drawAllTracking() {

	extern Mat finalTrackingFrame;
	extern vector<Point> detectedCoordinates;
	extern vector<Point> globalDetectedCoordinates;

	//cycle through all coordinates
	for (int v = 0; v < detectedCoordinates.size(); v++) {
		//save all detected coordinates
		globalDetectedCoordinates.push_back(detectedCoordinates[v]);

		//draw coordinates
		circle(finalTrackingFrame, detectedCoordinates[v], 4,
				Scalar(254, 254, 0), -1, 8, 0);
	}
	displayFrame("All Tracking Frame", finalTrackingFrame, true);
}



