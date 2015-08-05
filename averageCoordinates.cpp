/*
 * averageCoordinates.cpp
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

#include "drawCoordinates.h"
#include "sortCoordinates.h"
#include "averagePoints.h"

//defining contant PI
#define PI 3.14159265

//namespaces for convenience
using namespace cv;
using namespace std;

const int xDistanceThreshold = 450;
const int yDistanceThreshold = 75;

//recursive definition for clustering car points
vector <Point> averageCoordinatesDefinition(vector <Point> coordinates)
{
		//if more than 1 point
		if (coordinates.size() > 1)
		{
			//vectors of points
			vector<Point> destinationCoordinates;
			vector<Point> pointsToAverage;

			//sort coordinates
			coordinates = sortCoordinates(coordinates);

			//saving tmp point
			Point tmpPoint = coordinates[0];

			//control boolean
			bool enteredOnce = false;

			//cycling through all coordinates
			for (int v = 0; v < coordinates.size(); v++) {

				//if distance is above threshold
				if ((abs(tmpPoint.x - coordinates[v].x) > xDistanceThreshold) || ( abs(tmpPoint.y - coordinates[v].y) > yDistanceThreshold))
				{
					//save averaged coordinates
	 				destinationCoordinates.push_back(averagePoints(pointsToAverage));

	 				//read new tmp point
					tmpPoint = coordinates[v];

					//erase vector of points to average
					pointsToAverage.erase(pointsToAverage.begin(), pointsToAverage.end());

					//control boolean
					bool enteredOnce = true;
				}

				//if distance is below threshold
				else
				{
					//begin filling points to average
	 				pointsToAverage.push_back(coordinates[v]);
				}
			}

			//if not entered once
			if (!enteredOnce) {
				//average all coordinates
				destinationCoordinates.push_back(averagePoints(pointsToAverage));
			}

			//if only 1 point
			else if (pointsToAverage.size() == 1) {
				//save point
				destinationCoordinates.push_back(pointsToAverage[0]);
			}

			//if more than 1 point
			else if (pointsToAverage.size() > 0) {
				//average points
				destinationCoordinates.push_back(averagePoints(pointsToAverage));
			}

			//return processed coordinates
			return destinationCoordinates;
		}

		/*
		//if 1 coordinate
		else if (coordinates.size() > 0) {
 			//return coordinates
			return coordinates;
		}

		//if no coordinates
		else {
 			//create point
			coordinates.push_back(Point(0, 0));

			//return vector
			return coordinates;
		}
		*/

		else
		{
			return coordinates;
		}
}

vector <Point> recurseAverageCoordinates(	vector <Point> coordinates, int reps)
{
	for(int v = 0; v < reps; v++)
	{
		coordinates  = averageCoordinatesDefinition(coordinates);
		drawCoordinates(coordinates, "avgCoordinates" + to_string(v));
 	}
	return coordinates;
}

//average car points
vector<Point> averageCoordinates(vector<Point> coordinates, int distanceThreshold) {
	return recurseAverageCoordinates(coordinates, 1);
}



