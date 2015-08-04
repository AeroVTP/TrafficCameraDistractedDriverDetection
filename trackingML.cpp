/*
 * trackingML.cpp
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
#include "drawTmpTracking.h"
#include "drawAllTracking.h"

//namespaces for convenience
using namespace cv;
using namespace std;

//method to handle all tracking Machine Learning commands
void trackingML()
{
	extern int i;
	extern int bufferMemory;
	extern int mlBuffer;
	extern vector<Point> detectedCoordinates;

	//if CV is still initializing
	if (i <= bufferMemory + mlBuffer) {
		//display welcome image
		welcome(
				"Final Initialization; Running ML Startup -> Frames Remaining: "
						+ to_string((bufferMemory + mlBuffer + 1) - i));
	}

	//if ready to run
	else if (i > bufferMemory + mlBuffer + 1) {

		//if booting ML
		if (i == bufferMemory + mlBuffer + 1) {
			//display bootup message
			welcome("Initialization Complete -> Starting ML");
		}

		/*
		//begin processing frames
		else if (i > bufferMemory + mlBuffer + 1) {
			//display ready to run
			String tmpToDisplay = "Running ML Tracking -> Frame Number: "
					+ to_string(i);

			//display welcome image
			welcome(tmpToDisplay);
		}
		*/

		//process coordinates and average
		processCoordinates();

		//tracking all individual cars
		individualTracking();

		//draw coordinates in the tmp
		drawTmpTracking();

		//draw all car points
		drawAllTracking();
	}

	//erase detected coordinates for next run
	detectedCoordinates.erase(detectedCoordinates.begin(), detectedCoordinates.end());
}


