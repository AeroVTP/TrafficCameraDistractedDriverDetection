/*
 * fillCoordinates.cpp
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

//method to fill coordinates
void fillCoordinates(vector<Point2f> detectedCoordinatesMoments, int xLimiter, int yLimiter) {

	extern vector<Mat> globalGrayFrames;
	extern int i;
	extern vector<Point> detectedCoordinates;

	//cycle through all center points
	for (int v = 0; v < detectedCoordinatesMoments.size(); v++) {
		//creating tmp piont for each detected coordinate
		Point tmpPoint((int) detectedCoordinatesMoments[v].x,
				(int) detectedCoordinatesMoments[v].y);

		//if not in border
		if ((tmpPoint.x > xLimiter && tmpPoint.x < globalGrayFrames[i].cols - xLimiter)
				&& (tmpPoint.y > yLimiter
						&& tmpPoint.y < globalGrayFrames[i].rows - yLimiter)) {
			//saving into detected coordinates
			detectedCoordinates.push_back(tmpPoint);
		}
	}
}



