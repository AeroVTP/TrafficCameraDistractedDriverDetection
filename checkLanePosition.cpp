/*
 * checkLanePosition.cpp
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

//method to determine which lane is being driven through
int checkLanePosition(Point tmpPoint)
{
	extern vector<int> lanePositions;

	//if above first lane
	if(lanePositions[0] > tmpPoint.y)
	{
		//return in frame 0
		return 0;
	}

	//cycling through lanes
	for(int v =  1; v < lanePositions.size(); v++)
	{
		//finding which lane boundaries in
		if(lanePositions[v] > tmpPoint.y && lanePositions[v-1] < tmpPoint.y)
		{
			//return lane number
			return v;
		}
	}
}


