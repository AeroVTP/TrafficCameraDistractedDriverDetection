/*
 * registerFirstCar.cpp
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

//namespaces for convenience
using namespace cv;
using namespace std;

//register first car
void registerFirstCar() {

	//vector<vector<Point> > carCoordinates;
	extern vector<vector<Point> > vectorOfDetectedCars;
	extern vector<Point> detectedCoordinates;
	extern int FRAME_WIDTH;
	extern vector<vector<Point> > carCoordinates;
	extern int xLimiter;

	//save all cars
	vectorOfDetectedCars.push_back(detectedCoordinates);

	//cycling through points
	for (int v = 0; v < detectedCoordinates.size(); v++) {

		//if in the starting area on either side
		if(detectedCoordinates[v].x < (xLimiter * 1.5) || detectedCoordinates[v].y > FRAME_WIDTH - (xLimiter * 1.5)){
 			//creating vector of car coordinates
			vector<Point> carCoordinate;

			//saving car coordinates
			carCoordinate.push_back(detectedCoordinates[v]);
			carCoordinates.push_back(carCoordinate);
		}
	}
}




