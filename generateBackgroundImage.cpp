/*
 * generateBackgroundImage.cpp
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

//namespaces for convenience
using namespace cv;
using namespace std;

//method to handle all background image generation
void generateBackgroundImage(int FRAME_RATE) {

	extern bool readMedianImg;
	extern bool useMedians;
	extern int bufferMemory;
	extern int i;
	extern Mat backgroundFrameMedian;
	extern Mat drawAnomalyCar;
	extern Mat backgroundFrameMedianColor;
	extern Mat finalTrackingFrame;
	extern int medianImageCompletion;

	//if post-processing
	if (readMedianImg && useMedians && i < bufferMemory + 5) {
		extern string medianImageFilename;

		//read median image
		backgroundFrameMedian = imread("assets/" + medianImageFilename);

		//saving background to image
		backgroundFrameMedian.copyTo(drawAnomalyCar);
		backgroundFrameMedian.copyTo(backgroundFrameMedianColor);

		//convert to grayscale
		cvtColor(backgroundFrameMedian, backgroundFrameMedian, CV_BGR2GRAY);

		displayFrame("backgroundFrameMedian", backgroundFrameMedian);

		//saving background to image
		backgroundFrameMedian.copyTo(finalTrackingFrame);
	}

	//if real-time calculation
	else {
		//after initial buffer read and using medians
		if (i == bufferMemory && useMedians) {
			grayScaleFrameMedian();

			while (medianImageCompletion != 1) {
			}
		}
		//every 3 minutes
		if (i % (FRAME_RATE * 180) == 0 && i > 0) {
			//calculate new medians
			grayScaleFrameMedian();

			while (medianImageCompletion != 1) {
			}

		}
	}
	//signal completion
	medianImageCompletion = 0;
}


