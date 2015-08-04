/*
 * displayCoordinate.cpp
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
#include "welcome.h"
#include "checkBasicXYAnomaly.h"
#include "anomalyHandler.h"

using namespace std;
using namespace cv;

//display individual coordinate
void displayCoordinate(Point coordinate) {
	//display coordinate with formatting
	cout << "(" << coordinate.x << "," << coordinate.y << ")" << endl;
}

