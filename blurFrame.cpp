/*
 * blurFrame.cpp
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
 #include "blurFrame.h"

//namespaces for convenience
using namespace cv;
using namespace std;

//method to blur Mat using custom kernel size
Mat blurFrame(string blurType, Mat sourceDiffFrame, int blurSize) {
	extern bool debug;

	//Mat to hold blurred frame
	Mat blurredFrame;

	//if gaussian blur
	if (blurType == "gaussian") {
		//blur frame using custom kernel size
		blur(sourceDiffFrame, blurredFrame, Size(blurSize, blurSize),
				Point(-1, -1));

		//return blurred frame
		return blurredFrame;
	}

	//if blur type not implemented
	else {
		//report not implemented
		if (debug)
			cout << blurType << " type of blur not implemented yet" << endl;

		//return original frame
		return sourceDiffFrame;
	}

}



