/*
 * morphology.cpp
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

//namespaces for convenience
using namespace cv;
using namespace std;

//method to apply morphology
Mat morph(Mat sourceFrame, int amplitude, string type) {
	extern bool debug;

	//using default values
	double morph_size = .5;

	//performing two iterations
	const int iterations = 2;

	//constructing manipulation Mat
	Mat element = getStructuringElement(MORPH_RECT,
			Size(2 * morph_size + 1, 2 * morph_size + 1),
			Point(morph_size, morph_size));

	//if performing morphological closing
	if (type == "closing") {
		//repeat for increased effect
		for (int v = 0; v < amplitude; v++) {
			morphologyEx(sourceFrame, sourceFrame, MORPH_CLOSE, element,
					Point(-1, -1), iterations, BORDER_CONSTANT,
					morphologyDefaultBorderValue());
		}
	}

	//if performing morphological opening
	else if (type == "opening") {
		for (int v = 0; v < amplitude; v++) {
			//repeat for increased effect
			morphologyEx(sourceFrame, sourceFrame, MORPH_OPEN, element,
					Point(-1, -1), iterations, BORDER_CONSTANT,
					morphologyDefaultBorderValue());

		}
	}

	else if (type == "erode") {
		erode(sourceFrame, sourceFrame, element);
	}

	//if performing morphological gradient
	else if (type == "gradient") {
		//repeat for increased effect
		for (int v = 0; v < amplitude; v++) {
			morphologyEx(sourceFrame, sourceFrame, MORPH_GRADIENT, element,
					Point(-1, -1), iterations, BORDER_CONSTANT,
					morphologyDefaultBorderValue());
		}
	}

	//if performing morphological tophat
	else if (type == "tophat") {
		//repeat for increased effect
		for (int v = 0; v < amplitude; v++) {
			morphologyEx(sourceFrame, sourceFrame, MORPH_TOPHAT, element,
					Point(-1, -1), iterations, BORDER_CONSTANT,
					morphologyDefaultBorderValue());
		}
	}

	//if performing morphological blackhat
	else if (type == "blackhat") {
		//repeat for increased effect
		for (int v = 0; v < amplitude; v++) {
			morphologyEx(sourceFrame, sourceFrame, MORPH_BLACKHAT, element,
					Point(-1, -1), iterations, BORDER_CONSTANT,
					morphologyDefaultBorderValue());
		}
	}

	//if current morph operation is not availble
	else {
		//report cannot be done
		if (debug)
			cout << type << " type of morphology not implemented yet" << endl;
	}

	//return edited frame
	return sourceFrame;
}


