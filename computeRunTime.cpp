/*
 * computeRunTime.cpp
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

//namespaces for convenience
using namespace cv;
using namespace std;

//method to calculate runtime
void computeRunTime(clock_t t1, clock_t t2, int framesRead) {
	//subtract from start time
	float diff((float) t2 - (float) t1);

	//calculate frames per second
	double frameRateProcessing = (framesRead / diff) * CLOCKS_PER_SEC;

	//display amount of time for run time
	cout << (diff / CLOCKS_PER_SEC) << " seconds of run time." << endl;

	//display number of frames processed per second
	cout << frameRateProcessing << " frames processed per second." << endl;
	cout << framesRead << " frames read." << endl;
}


