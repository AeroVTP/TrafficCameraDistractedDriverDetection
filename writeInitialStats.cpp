/*
 * writeInitialStats.cpp
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

//namespaces for convenience
using namespace cv;
using namespace std;

//write initial statistics about the video
void writeInitialStats(int NUMBER_OF_FRAMES, int FRAME_RATE, int FRAME_WIDTH,
		int FRAME_HEIGHT, const char* filename) {
	extern bool debug;

	////writing stats to txt file
	//initiating write stream
	ofstream writeToFile;

	//creating filename  ending
	string filenameAppend = "Stats.txt";

	//concanating and creating file name string
	string strFilename = filename + currentDateTime() + filenameAppend;

	//open file stream and begin writing file
	writeToFile.open(strFilename);

	//write video statistics
	writeToFile << "Stats on video >> There are = " << NUMBER_OF_FRAMES
			<< " frames. The frame rate is " << FRAME_RATE
			<< " frames per second. Resolution is " << FRAME_WIDTH << " X "
			<< FRAME_HEIGHT;

	//close file stream
	writeToFile.close();

	if (debug) {
		//display video statistics
		cout << "Stats on video >> There are = " << NUMBER_OF_FRAMES
				<< " frames. The frame rate is " << FRAME_RATE
				<< " frames per second. Resolution is " << FRAME_WIDTH << " X "
				<< FRAME_HEIGHT << endl;
	}
}


