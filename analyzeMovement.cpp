/*
 * analyzeMovement.cpp
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

//defining contant PI
#define PI 3.14159265

//method to analyze all movement
void analyzeMovement()
{
	//setting y displacement threshold
	const int yDisplacementThreshold = 20;

	//booleans to detect anomalies
	bool detected = false;
	bool finishing = false;

	extern vector<vector<Point> > vectorOfDetectedCars;
	extern Mat backgroundFrameMedianColor;
	extern double xLearnedMovement;
	extern double yLearnedMovement;
	extern double learnedSpeedAverage;
	extern double learnedAngle;
	extern vector<double> distanceFromNormal;
	extern vector<Point> distanceFromNormalPoints;
	extern double xAverageMovement;
	extern double xDisplacement;
	extern double yDisplacement;
	extern double yLearnedMovement;
	extern double yAverageMovement;
	extern double yAverageCounter;
 	extern double learnedSpeed;
    extern double currentSpeed;
	extern double learnedDistanceAverage;
	extern double learnedDistance;
	extern double learnedDistanceCounter;
	extern double currentDistance;
	extern Mat distanceFrameAnomaly;
	extern double xAverageCounter;
	extern double learnedAggregate;
	extern double currentAngle;
	extern int numberOfAnomaliesDetected;
	extern Mat drawAnomalyCar;
	extern int i;
	extern int bufferMemory;
	extern int mlBuffer;
	extern Point tmpDetectPoint;
	extern int FRAME_WIDTH;

	//vector of current and previous detects
	vector<Point> currentDetects = vectorOfDetectedCars[vectorOfDetectedCars.size() - 1];
	vector<Point> prevDetects = vectorOfDetectedCars[vectorOfDetectedCars.size()- 2];

	//frames to draw movement properties
	Mat drawCoordinatesOnFrame;
	Mat drawCoordinatesOnFrameXY;
	Mat drawCoordinatesOnFrameSpeed;

	//saving background frame to draw on
	backgroundFrameMedianColor.copyTo(drawCoordinatesOnFrame);
	backgroundFrameMedianColor.copyTo(drawCoordinatesOnFrameXY);
	backgroundFrameMedianColor.copyTo(drawCoordinatesOnFrameSpeed);

	//draw learned models
	putText(drawCoordinatesOnFrameXY,
		(to_string(xLearnedMovement) + "|"
				+ to_string(yLearnedMovement)), Point(0, 30),
		CV_FONT_HERSHEY_SIMPLEX, 1, cvScalar(0, 0, 255), 1,
		CV_AA, false);

	putText(drawCoordinatesOnFrameSpeed,
		to_string(learnedSpeedAverage), Point(0, 30),
		CV_FONT_HERSHEY_SIMPLEX, 1, cvScalar(255, 255, 0), 1,
		CV_AA, false);

	putText(drawCoordinatesOnFrame, to_string(learnedAngle),
			Point(0, 30), CV_FONT_HERSHEY_SIMPLEX, 1,
			cvScalar(0, 0, 255), 1, CV_AA, false);

	//creating distance threshold
	const int distanceThreshold = 50;

	//finding almost number of detects
	int least = findMin(currentDetects.size(), prevDetects.size());

	//cycling through detects
	for (int v = 0; v < least; v++)
	{
		//sort detected coordinates
		currentDetects = sortCoordinates(currentDetects);
		prevDetects = sortCoordinates(prevDetects);

		//creating lowest distance with largest number
		double lowestDistance = INT_MAX;

		//variable to store current distance
		double distance;

		//creating point objects
		Point tmpPoint;
		Point tmpDetectPoint;
		Point bestPoint;

		//cycling through detects
		for (int j = 0; j < prevDetects.size(); j++)
		{
			//saving points
			tmpDetectPoint = prevDetects[j];
			tmpPoint = currentDetects[v];

			//calculating distance
			distance = sqrt(abs(tmpDetectPoint.x - tmpPoint.x)
					* (abs(tmpDetectPoint.y - tmpPoint.y)));

			//if distance is less than previous lowest distance
			if (distance < lowestDistance) {
				//resetting lowest distance
				lowestDistance = distance;

				//saving point which matches the best
				bestPoint = tmpDetectPoint;
			}
 		}

 		//determinig displacements
		int xDisplacement = abs(bestPoint.x - tmpPoint.x);
		int yDisplacement = abs(bestPoint.y - tmpPoint.y);

		//creating frame to display all anomalies
		Mat distanceFrameAnomaly;

		//saving median image background
		backgroundFrameMedianColor.copyTo(distanceFrameAnomaly);

		//cycling through all coordinates
		for (int d = 0; d < distanceFromNormal.size(); d++)
		{
			//sum all distances
			learnedDistanceCounter++;
			currentDistance = distanceFromNormal[d];
			learnedDistance += distanceFromNormal[d];

			//string to display current distance
			String movementStr = to_string(currentDistance);

			//write anomalies to frame
			putText(distanceFrameAnomaly, movementStr,
					distanceFromNormalPoints[d], CV_FONT_HERSHEY_SIMPLEX, 1,
					cvScalar(254, 254, 0), 1, CV_AA, false);

			//write learned distance
			putText(distanceFrameAnomaly, to_string(learnedDistanceAverage),
					Point(0, 30), CV_FONT_HERSHEY_SIMPLEX, 1,
					cvScalar(254, 254, 0), 1, CV_AA, false);
		}

		//erase vectors for next run
		distanceFromNormal.erase(distanceFromNormal.begin(), distanceFromNormal.end());
		distanceFromNormalPoints.erase(distanceFromNormalPoints.begin(), distanceFromNormalPoints.end());

		displayFrame("distanceFrameAnomaly", distanceFrameAnomaly, true);

		//average all learned distances
		learnedDistanceAverage = learnedDistance / learnedDistanceCounter;

		//if points are matched
		if (lowestDistance < distanceThreshold && yDisplacement < yDisplacementThreshold)
		{
			//suming all displacements
			xAverageMovement += xDisplacement;
			yAverageMovement += yDisplacement;

			//adding to counter
			xAverageCounter++;
			yAverageCounter++;

			//averaging movement
			xLearnedMovement = (xAverageMovement / xAverageCounter);
			yLearnedMovement = (yAverageMovement / yAverageCounter);

			//determining current speed
			currentSpeed = sqrt(xDisplacement * yDisplacement);

			//adding to total speed
			learnedSpeed += currentSpeed;

			//saving average speed
			learnedSpeedAverage = learnedSpeed / xAverageCounter;

			//calculating current angle
			double currentAngle = ((atan((double) yDisplacement) / (double) xDisplacement)) * 180 / PI;

			//saving angle into float
			float currentAngleFloat = (float) currentAngle;

			//if angle is more than 360
			if (currentAngle > 360)
			{
				//set angle as 0
				currentAngle = 0;
 			}

 			//if it is invalid
			if (currentAngleFloat != currentAngleFloat)
			{
				//set angle as 0
				currentAngle = 0;
 			}

 			//aggreage total angle
			learnedAggregate += currentAngle;

			//average angle
			learnedAngle = learnedAggregate / xAverageCounter;

			//if too fast
			if (learnedSpeedAverage * 3 < currentSpeed)
			{
				//create current speed
				String movementStr = to_string(currentSpeed);

				//write to frame
				putText(drawCoordinatesOnFrameSpeed, movementStr, bestPoint,
						CV_FONT_HERSHEY_SIMPLEX, 1, cvScalar(0, 0, 255), 1,
						CV_AA, false);

				//flag is detected
				detected = true;

				//aggregate number of anomalies detected
				numberOfAnomaliesDetected++;

				displayFrame("Speed Movement", drawCoordinatesOnFrameSpeed,
						true);
			}

			//if normal speed
			else
			{
				//create current speed
				String movementStr = to_string(currentSpeed);

				//write to frame
				putText(drawCoordinatesOnFrameSpeed, movementStr, bestPoint,
						CV_FONT_HERSHEY_SIMPLEX, 1, cvScalar(255, 255, 0), 1,
						CV_AA, false);

				displayFrame("Speed Movement", drawCoordinatesOnFrame, true);
			}

			if (currentAngle > 15)
			{
				//create current angle
				String movementStr = to_string(currentAngle);

				//write to frame
				putText(drawCoordinatesOnFrame, movementStr, bestPoint,
						CV_FONT_HERSHEY_SIMPLEX, 1, cvScalar(0, 0, 255), 1,
						CV_AA, false);

				//flag is detected
				detected = true;

				//aggregate number of anomalies detected
				numberOfAnomaliesDetected++;

				displayFrame("Angular Movement", drawCoordinatesOnFrame, true);
			}
			else
			{
				//create current angle
				String movementStr = to_string(currentAngle);

				//write to frame
				putText(drawCoordinatesOnFrame, movementStr, bestPoint,
						CV_FONT_HERSHEY_SIMPLEX, 1, cvScalar(255, 255, 0), 1,
						CV_AA, false);

				displayFrame("Angular Movement", drawCoordinatesOnFrame, true);
			}

			//is moving to much in the y direction
			if (yDisplacement > yLearnedMovement * 3)
			{
				//create current x y displacement
				String movementStr = to_string(xDisplacement) + "|"
						+ to_string(yDisplacement);

				//write to frame
				putText(drawCoordinatesOnFrameXY, movementStr, bestPoint,
										CV_FONT_HERSHEY_SIMPLEX, 1, cvScalar(255, 255, 0), 1,
										CV_AA, false);

				displayFrame("XY Movement", drawCoordinatesOnFrameXY, true);

				//draw anomaly
				circle(drawAnomalyCar, bestPoint, 5, Scalar(0, 0, 255), -1);

				//string to display
				String tmpToDisplay =
						"ANOMALY DETECTED (ANGLE) -> Frame Number: "
								+ to_string(i);

				//display welcome
				welcome(tmpToDisplay);

				//display anomaly message
				cout << " !!!!!!!!!!!ANOMALY DETECTED (ANGLE)!!!!!!!!!!!"
						<< endl;

				//flag is detected
				detected = true;

				//aggregate number of anomalies detected
				numberOfAnomaliesDetected++;

				displayFrame("Anomaly Car Detect Frame", drawAnomalyCar, true);
 			}

			else
			{
				//create current x y displacement
				String movementStr = to_string(xDisplacement) + "|"
						+ to_string(yDisplacement);

				//write to frame
				putText(drawCoordinatesOnFrameXY, movementStr, bestPoint,
						CV_FONT_HERSHEY_SIMPLEX, 1, cvScalar(0, 0, 255), 1,
						CV_AA, false);

				displayFrame("XY Movement", drawCoordinatesOnFrameXY, true);
			}

			//if past buffer memory, and inside the tracking area
			if (i > (bufferMemory + mlBuffer + 3) && tmpDetectPoint.x < FRAME_WIDTH - 30)
			{
				//check if anomalous
				checkBasicXYAnomaly(xDisplacement, yDisplacement,
						tmpDetectPoint, currentAngle);
			}
		}
	}

	//start anomaly handler
	anomalyHandler(detected, false);
}

