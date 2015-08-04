/*
 * drawTmpTracking.cpp
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

using namespace std;
using namespace cv;

#include "displayFrame.h"

//draw tmp history
void drawTmpTracking() {

	extern vector <Mat> globalFrames;
	extern int i;
	extern int numberOfCars;
	extern vector<vector<Point> > coordinateMemory;

	//creating memory for tracking
	const int thresholdPointMemory = 30 * numberOfCars;

	//creating counter
	int counter = coordinateMemory.size() - thresholdPointMemory;

	//creating tmp frame
	Mat tmpTrackingFrame;

	//saving frame
	globalFrames[i].copyTo(tmpTrackingFrame);

	//if counter is less than 0
	if (counter < 0) {

		//save as zero
 		counter = 0;
	}

	//cycling through coordinates
	for (int v = counter; v < coordinateMemory.size(); v++)
	{
		//moving through memory
		for (int j = 0; j < coordinateMemory[v].size(); j++)
		{
			//drawing circle
			circle(tmpTrackingFrame, coordinateMemory[v][j], 4,
					Scalar(254, 254, 0), -1, 8, 0);
		}
	}
	displayFrame("Tmp Tracking Frame", tmpTrackingFrame, true);
}




