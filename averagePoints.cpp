/*
 * averagePoints.cpp
 *
 *  Created on: Aug 5, 2015
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

#include "drawCoordinates.h"
#include "sortCoordinates.h"
#include "averagePoints.h"

//defining contant PI
#define PI 3.14159265

//namespaces for convenience
using namespace cv;
using namespace std;

//average multiple points
Point averagePoints(vector<Point> coordinates) {

	//if there is more than 1 coordinate
	if (coordinates.size() > 1) {

 		//variables to sum x and y coordinates
		double xCoordinate = 0;
		double yCoordinate = 0;

		//cycling through all coordinates and summing
		for (int v = 0; v < coordinates.size(); v++) {
			xCoordinate += coordinates[v].x;
			yCoordinate += coordinates[v].y;
		}

		//creating average point
		Point tmpPoint(xCoordinate / coordinates.size(),
				yCoordinate / coordinates.size());
		//returning average point
		return tmpPoint;
	}

	//if one point
	else if (coordinates.size() == 1) {
		//cout << "ONLY  POINT " << coordinates.size() << endl;

		//return 1 point
		return coordinates[0];
	}

	//if no points
	else {
		//cout << "ONLY POINT " << coordinates.size() << endl;

		//create point
		return Point(0, 0);
	}
}


