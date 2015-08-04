/*
 * displayFrame.cpp
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

//method to display frame
void displayFrame(string filename, Mat matToDisplay) {
	extern bool debug;
	//if in debug mode and Mat is not empty
	if (debug && matToDisplay.size[0] != 0) {
		imshow(filename, matToDisplay);
	}

	else if (matToDisplay.size[0] == 0) {
		cout << filename << " is empty, cannot be displayed." << endl;
	}
}

//method to display frame overriding debug
void displayFrame(string filename, Mat matToDisplay, bool override)
{
	//if override and Mat is not emptys
	if (override && matToDisplay.size[0] != 0 && filename != "Welcome")
	{
		imshow(filename, matToDisplay);
	}
	else if (override && matToDisplay.size[0] != 0)
	{
		namedWindow(filename);
		imshow(filename, matToDisplay);
	}
	else if (matToDisplay.size[0] == 0)
	{
		cout << filename << " is empty, cannot be displayed." << endl;
	}
}


