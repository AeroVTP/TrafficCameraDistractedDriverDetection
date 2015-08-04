/*
 * slidingWindowNeighborDetector.cpp
 *
 *  Created on: Aug 4, 2015
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
#include "trackingML.h"
#include "displayCoordinates.h"
#include "processExit.h"
#include "computeRunTime.h"

#include "gaussianMixtureModel.h"

//namespaces for convenience
using namespace cv;
using namespace std;

//method to perform proximity density search to remove noise and identify noise
Mat slidingWindowNeighborDetector(Mat sourceFrame, int numRowSections,
		int numColumnSections) {
	//if using default num rows
	if (numRowSections == -1 || numColumnSections == -1) {
		//split into standard size
		numRowSections = sourceFrame.rows / 10;
		numColumnSections = sourceFrame.cols / 20;
	}

	//declaring percentage to calculate density
	double percentage = 0;

	//setting size of search area
	int windowWidth = sourceFrame.rows / numRowSections;
	int windowHeight = sourceFrame.cols / numColumnSections;

	//creating destination frame of correct size
	Mat destinationFrame = Mat(sourceFrame.rows, sourceFrame.cols, CV_8UC1);

	//cycling through pieces
	for (int v = windowWidth / 2; v <= sourceFrame.rows - windowWidth / 2;
			v++) {
		for (int j = windowHeight / 2; j <= sourceFrame.cols - windowHeight / 2;
				j++) {
			//variables to calculate density
			double totalCounter = 0;
			double detectCounter = 0;

			//cycling through neighbors
			for (int x = v - windowWidth / 2; x < v + windowWidth / 2; x++) {
				for (int k = j - windowHeight / 2; k < j + windowHeight / 2;
						k++) {
					//if object exists
					if (sourceFrame.at<uchar>(x, k) > 127) {
						//add to detect counter
						detectCounter++;
					}

					//count pixels searched
					totalCounter++;
				}
			}

			//prevent divide by 0 if glitch and calculate percentage
			if (totalCounter != 0)
				percentage = detectCounter / totalCounter;
			else
				cout << "sWND Counter Error" << endl;

			//if object exists flag it
			if (percentage > .25) {
				destinationFrame.at<uchar>(v, j) = 255;
			}

			//else set it to 0
			else {
				//sourceFrame.at<uchar>(v,j) = 0;
				destinationFrame.at<uchar>(v, j) = 0;
			}
		}
	}

	//return processed frame
	return destinationFrame;
}



