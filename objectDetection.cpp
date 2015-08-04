/*
 * objectDetection.cpp
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
#include "opticalFlowFarneback.h"

#include "vibeBackgroundSubtraction.h"

#include "mogDetection.h"
#include "mogDetection2.h"

#include "medianDetection.h"

//namespaces for convenience
using namespace cv;
using namespace std;

//method to handle all image processing object detection
void objectDetection(int FRAME_RATE) {

	extern int i;
	extern vector<Mat> globalFrames;
	extern int medianDetectionGlobalFrameCompletion;
	extern Mat medianDetectionGlobalFrame;
	extern int mogDetection1GlobalFrameCompletion;
	extern int vibeDetectionGlobalFrameCompletion;
	extern Mat vibeDetectionGlobalFrame;
	extern Mat mogDetection1GlobalFrame;
	extern int mogDetection2GlobalFrameCompletion;
	extern Mat mogDetection2GlobalFrame;
	extern int bufferMemory;
	extern int mlBuffer;

	if (i > bufferMemory + 1) {
				String tmpToDisplay = "Running Image Analysis -> Frame Number: "
						+ to_string(i);
				welcome(tmpToDisplay);
	}

	//saving processed frame
	Mat gmmDetection = gaussianMixtureModel();
	Mat ofaDetection = opticalFlowFarneback();

	//start thread handlers
	vibeBackgroundSubtractionThreadHandler(false);
	mogDetectionThreadHandler(false);
	mogDetection2ThreadHandler(false);
	medianDetectionThreadHandler(FRAME_RATE);

	//booleans to control finish
	bool firstTimeMedianImage = true;
	bool firstTimeVibe = true;
	bool firstTimeMOG1 = true;
	bool firstTimeMOG2 = true;
	bool enterOnce = true;

	//Mats to save frames
	Mat tmpMedian;
	Mat tmpVibe;
	Mat tmpMOG1;
	Mat tmpMOG2;

	//booleans to register finish
	bool finishedMedian = false;
	bool finishedVibe = false;
	bool finishedMOG1 = false;
	bool finishedMOG2 = false;

	//while not finished
	while (!finishedMedian || !finishedVibe || !finishedMOG1 || !finishedMOG2
			|| enterOnce) {
		//controlling entered once
		enterOnce = false;

		//controlling median detector
		if (firstTimeMedianImage && medianDetectionGlobalFrameCompletion == 1) {
			//saving global frame
			tmpMedian = medianDetectionGlobalFrame;
			displayFrame("medianDetection", tmpMedian);

			//report finished
			firstTimeMedianImage = false;
			finishedMedian = true;

			//display welcome
 			welcome("Median BckSub Set -> FN: " + to_string(i));
		}

		//controlling ViBe detector
		if (firstTimeVibe && vibeDetectionGlobalFrameCompletion == 1) {
			//saving global frame
			tmpVibe = vibeDetectionGlobalFrame;
			displayFrame("vibeDetection", tmpVibe);

			//report finished
			firstTimeVibe = false;
			finishedVibe = true;

			//display welcome
  			welcome("ViBe PDF Model Set -> FN: " + to_string(i));
		}

		//controlling MOG1 detector
		if (firstTimeMOG1 && mogDetection1GlobalFrameCompletion == 1) {
			//saving global frame
			tmpMOG1 = mogDetection1GlobalFrame;
			displayFrame("mogDetection1", tmpMOG1);

			//report finished
			firstTimeMOG1 = false;
			finishedMOG1 = true;

			//display welcome
   			welcome("MOG1 Set -> FN: " + to_string(i));

		}

		//controlling MOG2 detector
		if (firstTimeMOG2 && mogDetection2GlobalFrameCompletion == 1) {
			//saving global frame
			tmpMOG2 = mogDetection2GlobalFrame;
			displayFrame("mogDetection2", tmpMOG2);

			//report finished
			firstTimeMOG2 = false;
			finishedMOG2 = true;

			//display welcome menu
   			welcome("MOG2 Set -> FN: " + to_string(i));

		}
	}

	//resetting completion variables
	vibeDetectionGlobalFrameCompletion = 0;
	mogDetection1GlobalFrameCompletion = 0;
	mogDetection2GlobalFrameCompletion = 0;
	medianDetectionGlobalFrameCompletion = 0;

	//display frames
	displayFrame("vibeDetection", tmpVibe);
	displayFrame("mogDetection1", tmpMOG1);
	displayFrame("mogDetection2", tmpMOG2);
	displayFrame("medianDetection", tmpMedian);
	displayFrame("gmmDetection", gmmDetection);
	displayFrame("ofaDetection", ofaDetection);

	//display raw frame
	displayFrame("Raw Frame", globalFrames[i], true);

	//if image are created and finished
	if (i > (bufferMemory + mlBuffer - 1) && tmpMOG1.channels() == 3
			&& tmpMOG2.channels() == 3 && ofaDetection.channels() == 3
			&& tmpMedian.channels() == 3) {
		//if all images are created and finished
		if (i > bufferMemory * 3 + 2 && tmpMOG1.channels() == 3
				&& tmpMOG2.channels() == 3 && ofaDetection.channels() == 3
				&& tmpMedian.channels() == 3 && tmpVibe.channels() == 3
				&& gmmDetection.channels() == 3 && 1 == 2) {

			//create combined image
			Mat combined = tmpMOG1 + tmpMOG2 + ofaDetection + tmpMedian
					+ tmpVibe + gmmDetection;

			//display image
			displayFrame("Combined Contours", combined);

			//create beta for weighting
			double beta = (1.0 - .5);

			//add the weighted image
			addWeighted(combined, .5, globalFrames[i], beta, 0.0, combined);

			displayFrame("Overlay", combined, true);
		}
		//if only 4 images are finished
		else {
			//combine all images
			Mat combined = tmpMOG1 + tmpMOG2 + ofaDetection + tmpMedian;
			displayFrame("Combined Contours", combined);

			//create beta for weighting
			double beta = (1.0 - .5);

			//add the weighted image
			addWeighted(combined, .5, globalFrames[i], beta, 0.0, combined);
			displayFrame("Overlay", combined, true);
		}
	}

	//report sync issue
	else {
		cout << "Sync Issue" << endl;
	}

}



