/*
 * findMin.cpp
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

using namespace std;
using namespace cv;

//identfy minimum between numbers
int findMin(int num1, int num2) {

	//if num1 is lower
	if (num1 < num2) {
		//return lower number
		return num1;
	}
	//if num2 is lower
	else if (num2 < num1)
	{
		//return lower number
		return num2;
	}
	//if they are the same
	else
	{
		//return num1
		return num1;
	}
}



