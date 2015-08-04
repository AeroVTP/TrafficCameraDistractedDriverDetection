/*
 * welcome.cpp
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

#include "findMin.h"
#include "sortCoordinates.h"
#include "displayFrame.h"

using namespace std;
using namespace cv;

//display welcome message and splash screen
void welcome() {
	extern int bufferMemory;
	extern int i;

	//displaying splash during bootup
	if (i < bufferMemory * 2) {
		//reading welcome image
		Mat img = imread("assets/TCD3 Splash V2.png");

		//displaying text
		putText(img, "Initializing; V. Prasad 2015 All Rights Reserved",
				cvPoint(15, 30), CV_FONT_HERSHEY_SIMPLEX, 1,
				cvScalar(255, 255, 0), 1, CV_AA, false);

		//display welcome images
		displayFrame("Welcome", img, true);
	}

	//shutting after bootup
	else {
		//close welcome image
		destroyWindow("Welcome");
	}
}

//display welcome message and splash screen
void welcome(String text) {
	//displaying splash with text
	Mat img = imread("assets/TCD3 Splash V2.png");

	//writing text on images
	putText(img, text, cvPoint(15, 30), CV_FONT_HERSHEY_SIMPLEX, .8,
			cvScalar(255, 255, 0), 1, CV_AA, false);

	//display welcome images
	displayFrame("Welcome", img, true);
}



