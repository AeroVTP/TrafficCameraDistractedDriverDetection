/*
 * sortCoordinates.cpp
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

//namespaces for convenience
using namespace cv;
using namespace std;

//method to compare points
bool point_comparator(const Point2f &a, const Point2f &b) {
	//determining difference between distances of points
	return a.x * a.x + a.y * a.y < b.x * b.x + b.y * b.y;
}

//method to sort all coordinates
vector<Point> sortCoordinates(vector<Point> coordinates) {
	//sort using point_compartor
	sort(coordinates.begin(), coordinates.end(), point_comparator);

	//return sorted coordinates
	return coordinates;
}


