/*
 * scanFinishedAverageCoordinates.cpp
 *
 *  Created on: Aug 6, 2015
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

#include "averageCoordinates.h"
#include "checkLanePosition.h"
#include "analyzeMovement.h"
#include "displayFrame.h"
#include "welcome.h"
#include "displayCoordinate.h"
#include "processCoordinates.h"
#include "individualTracking.h"
#include "registerFirstCar.h"
#include "sortCoordinates.h"
#include "distanceCoordinates.h"

using namespace std;
using namespace cv;

bool scanFinishedAverageCoordinates(vector <Point> coordinates)
{
	if(coordinates.size() > 1)
	{
		const double distanceThreshold = 50;

		double minimumDistance = INT_MAX;
		coordinates = sortCoordinates(coordinates);

		for( int v = 0; v < coordinates.size() - 1; v++)
		{
			double distance = distanceCoordinates(coordinates[v], coordinates[v+1]);
			if(minimumDistance > distance)
			{
				minimumDistance = distance;
			}
		}

		if(minimumDistance > distanceThreshold)
		{
			return false;
		}
		else
		{
			return true;
		}
	}
	else
	{
		return true;
	}
}



