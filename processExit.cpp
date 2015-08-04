/*
 * processExit.cpp
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

#include "computeRunTime.h"

//namespaces for convenience
using namespace cv;
using namespace std;

//method to process exit of software
bool processExit(VideoCapture capture, clock_t t1, char keyboardClick)
{
	extern vector<Mat> globalFrames;

	//if escape key is pressed
	if (keyboardClick == 27)
	{
		//display exiting message
		cout << "Exiting" << endl;

		//compute total run time
		computeRunTime(t1, clock(), (int) capture.get(CV_CAP_PROP_POS_FRAMES));

		//delete entire vector
		globalFrames.erase(globalFrames.begin(), globalFrames.end());

		//report file finished writing
		cout << "Finished writing file, Goodbye." << endl;

		//exit program
		return true;
	}

	else
	{
		return false;
	}
}



