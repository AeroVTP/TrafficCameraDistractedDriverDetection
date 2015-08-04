/*
 * averageCoordinates.h
 *
 *  Created on: Aug 3, 2015
 *      Author: Vidur
 */

#ifndef AVERAGECOORDINATES_H_
#define AVERAGECOORDINATES_H_

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

//defining contant PI
#define PI 3.14159265

//namespaces for convenience
using namespace cv;
using namespace std;

bool point_comparator(const Point2f &a, const Point2f &b);
vector<Point> averageCoordinates(vector<Point> coordinates, int distanceThreshold);
vector<Point> sortCoordinates(vector<Point> coordinates);
Point averagePoints(vector<Point> coordinates);

#endif /* AVERAGECOORDINATES_H_ */
