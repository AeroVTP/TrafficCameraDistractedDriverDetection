/*
 * initializeMat.cpp
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
#include "objectDetection.h"

#include "slidingWindowNeighborDetector.h"
#include "cannyContourDetector.h"
#include "opticalFlowFarneback.h"
#include "opticalFlowAnalysisObjectDetection.h"

#include "currentDateTime.h"
#include "type2StrTest.h"
#include "morphology.h"

#include "writeInitialStats.h"
#include "calculateFPS.h"
#include "pollOFAData.h"

#include "initializeMat.h"

//namespaces for convenience
using namespace cv;
using namespace std;

//method to initalize Mats on startup
void initilizeMat() {

	extern int i;
	extern Ptr<BackgroundSubtractorGMG> backgroundSubtractorGMM ;
	extern int bufferMemory;
	extern vector<Mat> globalGrayFrames;
	extern Mat backgroundFrameMedian;
	extern vector<int> lanePositions;
	extern int FRAME_HEIGHT;
	extern vector<vector<Point> > learnedCoordinates;
	extern int FRAME_WIDTH;
	extern vector< vector <int> > accessTimesInt;

	const int numberOfLanes = 6;

	//if first run
	if (i == 0) {

		//initialize background subtractor object
		backgroundSubtractorGMM->set("initializationFrames", bufferMemory);
		backgroundSubtractorGMM->set("decisionThreshold", 0.85);

		//save gray value to set Mat parameters
		globalGrayFrames[i].copyTo(backgroundFrameMedian);

		//saving all lane positions
 		lanePositions.push_back(57);
		lanePositions.push_back(120);
		lanePositions.push_back(200);
		lanePositions.push_back(290);
		lanePositions.push_back(390);
		lanePositions.push_back(FRAME_HEIGHT - 1);

		//stepping through lane positions
		for (int j = 0; j < lanePositions.size(); j++) {

			//create tmp variables
			double tmpLanePosition = 0;
			double normalLanePosition = 0;

			//reading vector
			tmpLanePosition = lanePositions[j];

			//creating vectors to save points
			vector<Point> tmpPointVector;
			vector<Point> accessTimesFirstVect;
			vector<int> accessTimesFirstVectInt;
			vector <int> accessTimesVectorFirstLayer;

			//stepping through all LASM coordinates
			for (int v = 0; v < FRAME_WIDTH; v += 7) {
				//if first lane divide by 2s
				if (j == 0)
					normalLanePosition = tmpLanePosition / 2;

				//divide two lane positions
				else
					normalLanePosition = (tmpLanePosition + lanePositions[j - 1]) / 2;

				//save all vectors
				tmpPointVector.push_back(Point(v, normalLanePosition));
				accessTimesFirstVect.push_back(Point(1, 0));
				accessTimesFirstVectInt.push_back(0);
				accessTimesVectorFirstLayer.push_back(1);
			}

			//save all vectors into LASM model
			learnedCoordinates.push_back(tmpPointVector);

			accessTimesInt.push_back(accessTimesVectorFirstLayer);
		}
	}
}




