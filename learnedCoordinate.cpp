/*
 * learnedCoordinate.cpp
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

#include "averageCoordinates.h"
#include "checkLanePosition.h"
#include "analyzeMovement.h"
#include "displayFrame.h"
#include "welcome.h"
#include "displayCoordinate.h"
#include "processCoordinates.h"
#include "individualTracking.h"
#include "registerFirstCar.h"

#include "learnedDistanceFromNormal.h"

//LASM method
void learnedCoordinate()
{
	extern Mat backgroundFrameColorMedian;
	extern vector<Point> detectedCoordinates;
	extern vector<vector<Point> > learnedCoordinates;
	extern vector<int> lanePositions;
	extern int FRAME_WIDTH;
	extern vector<double> distanceFromNormal;
	extern vector<Point> distanceFromNormalPoints;
	extern vector< vector <int> > accessTimesInt;

	//Mat to show model
	Mat distanceFrame;

	//displaying color median to background
	backgroundFrameColorMedian.copyTo(distanceFrame);

	//variable to show lane
 	int initialCounter = 0;

 	//cycling through all coordinates
 	for (int v = 0; v < detectedCoordinates.size(); v++)
	{
		//saving tmp point
 		Point tmpPoint = detectedCoordinates[v];

 		//start at correct lane
		initialCounter = checkLanePosition(tmpPoint);

		//cycling through all of one lanes positions
		for (int j = 0; j < learnedCoordinates[initialCounter].size(); j++)
		{
			//reading old point
	    	Point existingPoint = learnedCoordinates[initialCounter][tmpPoint.x / 7];

	    	//average points
			Point averagedPoint(((existingPoint.x + tmpPoint.x) / 2),
					((existingPoint.y + tmpPoint.y) / 2));

			//writing to LASM
			learnedCoordinates[initialCounter][tmpPoint.x / 7] = averagedPoint;
			accessTimesInt[initialCounter][tmpPoint.x / 7] = accessTimesInt[initialCounter][tmpPoint.x / 7]  + 1;

			double averagedDistanceFromNormal =	sqrt(abs(existingPoint.x - tmpPoint.x) *
									abs(existingPoint.y - tmpPoint.y));

			//if LASM changed
			if(averagedDistanceFromNormal != 0)
			{
				learnedDistanceFromNormal(averagedDistanceFromNormal);

				//display difference
				//cout << " DIFFERENCE FROM NORMAL " << to_string(sqrt( abs(existingPoint.x - tmpPoint.x) *
				//	abs(existingPoint.y - tmpPoint.y))) << endl;
			}

		}

		//if ready to break
		bool breakNow = false;

		//cycling through lanes
		for (int j = 0; j < lanePositions.size(); j++)
		{
			//creating tmp lane positions
			double tmpLanePosition = 0;
			double normalLanePosition = 0;

			//saving current lane position
			tmpLanePosition = lanePositions[j];

			//creating vector to hold points
			vector<Point> tmpPointVector;

			//if correct lane
			if ((lanePositions[j] >= tmpPoint.y) && !breakNow)
			{
				//cycling through sectors
				for (int v = 0; v < FRAME_WIDTH; v += 7)
				{
					//determining sector points
					if ((v >= tmpPoint.x - 6 && v <= tmpPoint.x + 6)	&& !breakNow)
					{
						//reading out existing vector to edit
						tmpPointVector = learnedCoordinates[j];

						//reading out old unedited point
						Point oldTmpPoint = tmpPointVector.at(tmpPoint.x / 7);

						//tmp variable to eventually determine number of reads
						//int tmpATI = 2;

						int tmpATI = accessTimesInt[initialCounter][tmpPoint.x / 7];

						//determining average y
						//int tmp = ((tmpPoint.y +  oldTmpPoint.y)) / tmpATI;
						double averagedDistanceFromNormal = ((oldTmpPoint.y * tmpATI - 1) + tmpPoint.y) / tmpATI;

						//learnedDistanceFromNormal(averagedDistanceFromNormal);

						//learnedDistanceFromNormal(tmpPoint.y - tmpPointVector.at(tmpPoint.x / 7).y);

 						//write averaged values to vector
						distanceFromNormal.push_back(abs(	tmpPoint.y - tmpPointVector.at(tmpPoint.x / 7).y));
						distanceFromNormalPoints.push_back(tmpPoint);

						//writing to frame
						putText(distanceFrame, to_string(abs(tmpPoint.y - tmpPointVector.at(tmpPoint.x / 7).y)),
								tmpPoint, 3, 1, Scalar(254, 254, 0), 2);

						//drawing onto frame
						circle(distanceFrame, tmpPoint, 4, Scalar(254, 254, 0), -1, 8, 0);

						//create averaged points
						Point averagePoint(v, averagedDistanceFromNormal);

						//saving averaged point to vector
						tmpPointVector.at(tmpPoint.x / 7) = averagePoint;

						//saving back to LASM
						learnedCoordinates[j] = tmpPointVector;

						//break now that is finished
						breakNow = true;
					}

					//if ready to break
					if (breakNow)
					{
						//forward to edge case
						v = FRAME_WIDTH;
					}
				}
			}

			//if ready to break
			if (breakNow)
			{
				//forward to edge case
				j = lanePositions.size();
			}
		}
	}
	//displayFrame("distanceFrame", distanceFrame);
}


