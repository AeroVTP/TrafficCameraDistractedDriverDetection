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

/*
 * anomalyHandler.cpp
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

//anomaly handler to reduce false-positives
void anomalyHandler(bool detected, bool finishing)
{
	extern int lastAnomalyDetectedFN;
	extern int numberOfAnomaliesDetected;
	extern int i;

	//if not in exiting zone
	if(!finishing)
	{
		//if detected anomaly
		if(detected)
		{
			//last anomaly detected
			lastAnomalyDetectedFN = i;

			//if 10 anomalies detected
			if(numberOfAnomaliesDetected > 10)
			{
				//report final anomaly detected
				cout << "ANOMALY DETECTED" << endl;
				welcome("ANOMALY DETECTED");
			}
		}

		//if memory past last anomaly
		else if(lastAnomalyDetectedFN < i - 45)
		{
			//reset anomaly counter
			welcome("NORMAL OPS");
			numberOfAnomaliesDetected = 0;
		}
	}
}


/*
 * averageCoordinates.cpp
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

#include "sortCoordinates.h"

//defining contant PI
#define PI 3.14159265

//namespaces for convenience
using namespace cv;
using namespace std;

//average multiple points
Point averagePoints(vector<Point> coordinates) {

	//if there is more than 1 coordinate
	if (coordinates.size() > 1) {

 		//variables to sum x and y coordinates
		double xCoordinate = 0;
		double yCoordinate = 0;

		//cycling through all coordinates and summing
		for (int v = 0; v < coordinates.size(); v++) {
			xCoordinate += coordinates[v].x;
			yCoordinate += coordinates[v].y;
		}

		//creating average point
		Point tmpPoint(xCoordinate / coordinates.size(),
				yCoordinate / coordinates.size());
		//returning average point
		return tmpPoint;
	}

	//if one point
	else if (coordinates.size() == 1) {
		cout << "ONLY  POINT " << coordinates.size() << endl;

		//return 1 point
		return coordinates[0];
	}

	//if no points
	else {
		cout << "ONLY POINT " << coordinates.size() << endl;

		//create point
		return Point(0, 0);
	}
}


//average car points
vector<Point> averageCoordinates(vector<Point> coordinates, int distanceThreshold) {

	//if more than 1 point
	if (coordinates.size() > 1)
	{
		//vectors of points
		vector<Point> destinationCoordinates;
		vector<Point> pointsToAverage;

		//sort coordinates
		coordinates = sortCoordinates(coordinates);

		//saving tmp point
		Point tmpPoint = coordinates[0];

		//control boolean
		bool enteredOnce = false;

		//cycling through all coordinates
		for (int v = 0; v < coordinates.size(); v++) {

			//if distance is above threshold
			if (sqrt(
					(abs(tmpPoint.y - coordinates[v].y)
							* (abs(tmpPoint.x - coordinates[v].x))))
					> distanceThreshold) {

				//save averaged cordinates
 				destinationCoordinates.push_back(averagePoints(pointsToAverage));

 				//read new tmp point
				tmpPoint = coordinates[v];

				//erase vector of points to average
				pointsToAverage.erase(pointsToAverage.begin(), pointsToAverage.end());

				//control boolean
				bool enteredOnce = true;
			}

			//if distance is below threshold
			else
			{
				//begin filling points to average
 				pointsToAverage.push_back(coordinates[v]);
			}
		}

		//if not entered once
		if (!enteredOnce) {
			//average all coordinates
			destinationCoordinates.push_back(averagePoints(pointsToAverage));
		}

		//if only 1 point
		else if (pointsToAverage.size() == 1) {
			//save point
			destinationCoordinates.push_back(pointsToAverage[0]);
		}

		//if more than 1 point
		else if (pointsToAverage.size() > 0) {
			//average points
			destinationCoordinates.push_back(averagePoints(pointsToAverage));
		}

		//return processed coordinates
		return destinationCoordinates;
	}

	//if 1 coordinate
	else if (coordinates.size() > 0) {
		cout << " RETURNING 1 POINT COOORDINATE" << endl;

		//return coordinates
		return coordinates;
	}

	//if no coordinates
	else {
		cout << " FIRST EMPTY POINT MADE ONE UP " << endl;

		//create point
		coordinates.push_back(Point(0, 0));

		//return vector
		return coordinates;
	}
}



#include "bgfg_vibe.hpp"

bgfg_vibe::bgfg_vibe():R(20),N(20),noMin(2),phi(0)
{
    initDone=false;
    rnd=cv::theRNG();
    ri=0;
}
void bgfg_vibe::init()
{
    for(int i=0;i<rndSize;i++)
    {
        rndp[i]=rnd(phi);
        rndn[i]=rnd(N);
        rnd8[i]=rnd(8);
    }
}
void bgfg_vibe::setphi(int phi)
{
    this->phi=phi;
    for(int i=0;i<rndSize;i++)
    {
        rndp[i]=rnd(phi);
    }
}
void bgfg_vibe::init_model(cv::Mat& firstSample)
{
    std::vector<cv::Mat> channels;
    split(firstSample,channels);
    if(!initDone)
    {
        init();
        initDone=true;
    }
    model=new Model;
    model->fgch= new cv::Mat*[channels.size()];
    model->samples=new cv::Mat**[N];
    model->fg=new cv::Mat(cv::Size(firstSample.cols,firstSample.rows), CV_8UC1);
    for(size_t s=0;s<channels.size();s++)
    {
        model->fgch[s]=new cv::Mat(cv::Size(firstSample.cols,firstSample.rows), CV_8UC1);
        cv::Mat** samples= new cv::Mat*[N];
        for(int i=0;i<N;i++)
        {
            samples[i]= new cv::Mat(cv::Size(firstSample.cols,firstSample.rows), CV_8UC1);
        }
        for(int i=0;i<channels[s].rows;i++)
        {
            int ioff=channels[s].step.p[0]*i;
            for(int j=0;j<channels[0].cols;j++)
            {
                for(int k=0;k<1;k++)
                {
                    (samples[k]->data + ioff)[j]=channels[s].at<uchar>(i,j);
                }
                (model->fgch[s]->data + ioff)[j]=0;

                if(s==0)(model->fg->data + ioff)[j]=0;
            }
        }
        model->samples[s]=samples;
    }
}
void bgfg_vibe::fg1ch(cv::Mat& frame,cv::Mat** samples,cv::Mat* fg)
{
    int step=frame.step.p[0];
    for(int i=1;i<frame.rows-1;i++)
    {
        int ioff= step*i;
        for(int j=1;j<frame.cols-1;j++)
        {
            int count =0,index=0;
            while((count<noMin) && (index<N))
            {
                int dist= (samples[index]->data + ioff)[j]-(frame.data + ioff)[j];
                if(dist<=R && dist>=-R)
                {
                    count++;
                }
                index++;
            }
            if(count>=noMin)
            {
                ((fg->data + ioff))[j]=0;
                int rand= rndp[rdx];
                if(rand==0)
                {
                    rand= rndn[rdx];
                    (samples[rand]->data + ioff)[j]=(frame.data + ioff)[j];
                }
                rand= rndp[rdx];
                int nxoff=ioff;
                if(rand==0)
                {
                    int nx=i,ny=j;
                    int cases= rnd8[rdx];
                    switch(cases)
                    {
                    case 0:
                        //nx--;
                        nxoff=ioff-step;
                        ny--;
                        break;
                    case 1:
                        //nx--;
                        nxoff=ioff-step;
                        ny;
                        break;
                    case 2:
                        //nx--;
                        nxoff=ioff-step;
                        ny++;
                        break;
                    case 3:
                        //nx++;
                        nxoff=ioff+step;
                        ny--;
                        break;
                    case 4:
                        //nx++;
                        nxoff=ioff+step;
                        ny;
                        break;
                    case 5:
                        //nx++;
                        nxoff=ioff+step;
                        ny++;
                        break;
                    case 6:
                        //nx;
                        ny--;
                        break;
                    case 7:
                        //nx;
                        ny++;
                        break;
                    }
                    rand= rndn[rdx];
                    (samples[rand]->data + nxoff)[ny]=(frame.data + ioff)[j];
                }
            }else
            {
                ((fg->data + ioff))[j]=255;
            }
        }
    }
}
cv::Mat* bgfg_vibe::fg(cv::Mat& frame)
{
    std::vector<cv::Mat> channels;
    split(frame,channels);
    for(size_t i=0;i<channels.size();i++)
    {
        fg1ch(channels[i],model->samples[i],model->fgch[i]);
        if(i>0 && i<2)
        {
            bitwise_or(*model->fgch[i-1],*model->fgch[i],*model->fg);
        }
        if(i>=2)
        {
            bitwise_or(*model->fg,*model->fgch[i],*model->fg);
        }
    }
    if(channels.size()==1) return model->fgch[0];
    return model->fg;
}
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



/*
 * calcMedian.cpp
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

#include "grayScaleFrameMedian.h"

//namespaces for convenience
using namespace cv;
using namespace std;

//method to calculate median of vector of integers
double calcMedian(vector<int> integers) {
	//double to store non-int median
	double median;

	//read size of vector
	size_t size = integers.size();

	//sort array
	sort(integers.begin(), integers.end());

	//if even number of elements
	if (size % 2 == 0) {
		//median is middle elements averaged
		median = (integers[size / 2 - 1] + integers[size / 2]) / 2;
	}

	//if odd number of elements
	else {
		//median is middle element
		median = integers[size / 2];
	}

	//return the median value
	return median;
}



/*
 * calculateDeviance.cpp
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
#include "learnedCoordinate.h"

//method to calculate deviance
void calculateDeviance() {

	extern Mat backgroundFrameMedianColor;
	extern vector< vector<Point> > learnedCoordinates;

	//train LASM
	learnedCoordinate();

	//Mat to display LASM
	Mat tmpToDisplay;

	//save background
	backgroundFrameMedianColor.copyTo(tmpToDisplay);

	//cycle through LASM coordinates
	for (int v = 0; v < learnedCoordinates.size(); v++) {
		for (int j = 0; j < learnedCoordinates[v].size(); j++) {
			//read tmpPoint
			Point tmpPoint = learnedCoordinates[v][j];

			//create top left point
			Point tmpPointTopLeft(tmpPoint.x - .01, tmpPoint.y + .01);

			//create bottom right point
			Point tmpPointBottomRight(tmpPoint.x + .01, tmpPoint.y - .01);

			//create rectangle to display
	        rectangle( tmpToDisplay, tmpPointTopLeft, tmpPointBottomRight, Scalar(255, 255, 0), 1);
		}
	}
	displayFrame("Learned Path", tmpToDisplay, true);
}



/*
 * calculateFPS.cpp
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

//namespaces for convenience
using namespace cv;
using namespace std;


//calculate time for each iteration
double calculateFPS(clock_t tStart, clock_t tFinal) {
	//return frames per second
	return 1 / ((((float) tFinal - (float) tStart) / CLOCKS_PER_SEC));
}



/*
 * cannyContourDetector.cpp
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
#include "fillCoordinates.h"

//namespaces for convenience
using namespace cv;
using namespace std;

//method to draw canny contours
Mat cannyContourDetector(Mat srcFrame) {
	extern int FRAME_WIDTH;
	extern int FRAME_HEIGHT;

	//threshold for non-car objects or noise
	const int thresholdNoiseSize = 200;
	const int misDetectLargeSize = 600;

	//instantiating Mat and Canny objects
	Mat canny;
	Mat cannyFrame;
	vector<Vec4i> hierarchy;
	typedef vector<vector<Point> > TContours;
	TContours contours;

	//run canny edge detector
	Canny(srcFrame, cannyFrame, 300, 900, 3);
	findContours(cannyFrame, contours, hierarchy, CV_RETR_CCOMP,
			CV_CHAIN_APPROX_NONE);

	//creating blank frame to draw on
	Mat drawing = Mat::zeros(cannyFrame.size(), CV_8UC3);

	//moments for center of mass
	vector<Moments> mu(contours.size());
	for (int i = 0; i < contours.size(); i++) {
		mu[i] = moments(contours[i], false);
	}

	//get mass centers:
	vector<Point2f> mc(contours.size());
	for (int i = 0; i < contours.size(); i++) {
		mc[i] = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
	}

	//for each detected contour
	for (int v = 0; v < contours.size(); v++) {
		//if large enough to be object
		if (arcLength(contours[v], true) > thresholdNoiseSize
				&& arcLength(contours[v], true) < misDetectLargeSize) {
			if((mc[v].x > 30 && mc[v].x < FRAME_WIDTH - 60) && (mc[v].y > 30 && mc[v].y < FRAME_HEIGHT - 60))
			{
				//draw object and circle center point
				drawContours(drawing, contours, v, Scalar(254, 254, 0), 2, 8,
						hierarchy, 0, Point());
				circle(drawing, mc[v], 4, Scalar(254, 254, 0), -1, 8, 0);
			}
			fillCoordinates(mc);
		}
	}

	//return image with contours
	return drawing;
}


/*
 * checkBasicXYAnomaly.cpp
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

//namespaces for convenience
using namespace cv;
using namespace std;

//boolean to determine if anomaly is detected
bool checkBasicXYAnomaly(int xMovement, int yMovement, Point carPoint, double currentAngle) {

	//thresholds to determine if anomaly
	const double maxThreshold = 5;
	const double minThreshold = -5;
	const int maxMovement = 30;

	const int angleThresholdMax = 10;
	const int angleThresholdMin = -10;
	const int angleThreshold = 20;

	//if above angle threshold
	if (currentAngle > angleThreshold)
	{
		//write coordinate
		displayCoordinate(carPoint);
		cout << " !!!!!!!!!!!ANOMALY DETECTED (ANGLE)!!!!!!!!!!!" << endl;
		//is anomalous
		return true;
 	}
	else
 	{
 		//is normal
 		return false;
 	}
}


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


/*
 * computeRunTime.cpp
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

//namespaces for convenience
using namespace cv;
using namespace std;

//method to calculate runtime
void computeRunTime(clock_t t1, clock_t t2, int framesRead) {
	//subtract from start time
	float diff((float) t2 - (float) t1);

	//calculate frames per second
	double frameRateProcessing = (framesRead / diff) * CLOCKS_PER_SEC;

	//display amount of time for run time
	cout << (diff / CLOCKS_PER_SEC) << " seconds of run time." << endl;

	//display number of frames processed per second
	cout << frameRateProcessing << " frames processed per second." << endl;
	cout << framesRead << " frames read." << endl;
}


/*
 * currentDateTime.cpp
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

#include "grayScaleFrameMedian.h"
#include "calcMedian.h"
#include "currentDateTime.h"

//namespaces for convenience
using namespace cv;
using namespace std;

//method that returns date and time as a string to tag txt files
const string currentDateTime() {
	extern string fileTime;

	//creating time object that reads current time
	time_t now = time(0);

	//creating time structure
	struct tm tstruct;

	//creating a character buffer of 80 characters
	char buf[80];

	//checking current local time
	tstruct = *localtime(&now);

	//writing time to string
	strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

	fileTime = buf;

	//returning the string with the time
	return buf;
}


/*
 * displayCoordinate.cpp
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

//display individual coordinate
void displayCoordinate(Point coordinate) {
	//display coordinate with formatting
	cout << "(" << coordinate.x << "," << coordinate.y << ")" << endl;
}

/*
 * displayCoordinates.cpp
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

//method to display all coordinates
void displayCoordinates(vector<Point> coordinatesToDisplay) {
	//cycling through each coordinate
	for (int v = 0; v < coordinatesToDisplay.size(); v++) {
		//displaying coordinate
		cout << "(" << coordinatesToDisplay[v].x << ","
				<< coordinatesToDisplay[v].y << ")" << endl;
	}
}


/*
 * displayFrame.cpp
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

using namespace std;
using namespace cv;

//method to display frame
void displayFrame(string filename, Mat matToDisplay) {
	extern bool debug;
	//if in debug mode and Mat is not empty
	if (debug && matToDisplay.size[0] != 0) {
		imshow(filename, matToDisplay);
	}

	else if (matToDisplay.size[0] == 0) {
		cout << filename << " is empty, cannot be displayed." << endl;
	}
}

//method to display frame overriding debug
void displayFrame(string filename, Mat matToDisplay, bool override)
{
	//if override and Mat is not emptys
	if (override && matToDisplay.size[0] != 0 && filename != "Welcome")
	{
		imshow(filename, matToDisplay);
	}
	else if (override && matToDisplay.size[0] != 0)
	{
		namedWindow(filename);
		imshow(filename, matToDisplay);
	}
	else if (matToDisplay.size[0] == 0)
	{
		cout << filename << " is empty, cannot be displayed." << endl;
	}
}


/*
 * drawAllTracking.cpp
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

#include "displayFrame.h"

using namespace std;
using namespace cv;

//draw all coordinates
void drawAllTracking() {

	extern Mat finalTrackingFrame;
	extern vector<Point> detectedCoordinates;
	extern vector<Point> globalDetectedCoordinates;

	//cycle through all coordinates
	for (int v = 0; v < detectedCoordinates.size(); v++) {
		//save all detected coordinates
		globalDetectedCoordinates.push_back(detectedCoordinates[v]);

		//draw coordinates
		circle(finalTrackingFrame, detectedCoordinates[v], 4,
				Scalar(254, 254, 0), -1, 8, 0);
	}
	displayFrame("All Tracking Frame", finalTrackingFrame, true);
}



/*
 * drawCoordinates.cpp
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
#include "displayCoordinates.h"
#include "drawCoordinates.h"

//method to draw coordinates to frame
void drawCoordinates(vector<Point> coordinatesToDisplay, String initialName) {
	extern Mat backgroundFrameMedian;

	//mat to draw frame
	Mat tmpToDraw;

	//using background frame to write to
	backgroundFrameMedian.copyTo(tmpToDraw);

	//cycle through all coordinates
	for (int v = 0; v < coordinatesToDisplay.size(); v++) {
		//draw all coordinates
		circle(tmpToDraw, coordinatesToDisplay[v], 4, Scalar(254, 254, 0), -1,
				8, 0);
	}

 	displayFrame(initialName, tmpToDraw, true);
}

/*
 * drawTmpTracking.cpp
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

using namespace std;
using namespace cv;

#include "displayFrame.h"

//draw tmp history
void drawTmpTracking() {

	extern vector <Mat> globalFrames;
	extern int i;
	extern int numberOfCars;
	extern vector<vector<Point> > coordinateMemory;

	//creating memory for tracking
	const int thresholdPointMemory = 10 * numberOfCars;

	//creating counter
	int counter = coordinateMemory.size() - thresholdPointMemory;

	//creating tmp frame
	Mat tmpTrackingFrame;

	//saving frame
	globalFrames[i].copyTo(tmpTrackingFrame);

	//if counter is less than 0
	if (counter < 0) {

		//save as zero
 		counter = 0;
	}

	//cycling through coordinates
	for (int v = counter; v < coordinateMemory.size(); v++)
	{
		//moving through memory
		for (int j = 0; j < coordinateMemory[v].size(); j++)
		{
			//drawing circle
			circle(tmpTrackingFrame, coordinateMemory[v][j], 4,
					Scalar(254, 254, 0), -1, 8, 0);
		}
	}
	displayFrame("Tmp Tracking Frame", tmpTrackingFrame, true);
}




/*
 * fillCoordinates.cpp
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
#include "fillCoordinates.h"

//namespaces for convenience
using namespace cv;
using namespace std;

//method to fill coordinates
void fillCoordinates(vector<Point2f> detectedCoordinatesMoments) {

	extern vector<Mat> globalGrayFrames;
	extern int i;
	extern vector<Point> detectedCoordinates;

	//cycle through all center points
	for (int v = 0; v < detectedCoordinatesMoments.size(); v++) {
		//creating tmp piont for each detected coordinate
		Point tmpPoint((int) detectedCoordinatesMoments[v].x,
				(int) detectedCoordinatesMoments[v].y);

		//if not in border
		if ((tmpPoint.x > 30 && tmpPoint.x < globalGrayFrames[i].cols - 60)
				&& (tmpPoint.y > 30
						&& tmpPoint.y < globalGrayFrames[i].rows - 30)) {
			//saving into detected coordinates
			detectedCoordinates.push_back(tmpPoint);
		}
	}
}



/*
 * findMin.cpp
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

using namespace std;
using namespace cv;

//identfy minimum between numbers
int findMin(int num1, int num2) {

	//if num1 is lower
	if (num1 < num2) {
		//return lower number
		return num1;
	}
	//if num2 is lower
	else if (num2 < num1)
	{
		//return lower number
		return num2;
	}
	//if they are the same
	else
	{
		//return num1
		return num1;
	}
}



/*
 * gaussianMixtureModel.cpp
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

#include "slidingWindowNeighborDetector.h"
#include "cannyContourDetector.h"

using namespace std;
using namespace cv;

//defining format of data sent to threads
struct thread_data {
	//include int for data passing
	int data;
};

//method to calculate Gaussian image difference
void *calcGaussianMixtureModel(void *threadarg) {
	extern vector<Mat> globalFrames;
	extern int i;
	extern Mat gmmFrameRaw;
	extern Ptr<BackgroundSubtractorGMG> backgroundSubtractorGMM ;
	extern Mat binaryGMMFrame;
	extern Mat gmmTempSegmentFrame;
	extern Mat gmmFrame;
	extern int bufferMemory;
	extern Mat cannyGMM;
	extern int gaussianMixtureModelCompletion;
	//perform deep copy
	globalFrames[i].copyTo(gmmFrameRaw);

	//update model
	(*backgroundSubtractorGMM)(gmmFrameRaw, binaryGMMFrame);

	//save into tmp frame
	gmmFrameRaw.copyTo(gmmTempSegmentFrame);

	//add movement mask
	add(gmmFrameRaw, Scalar(0, 255, 0), gmmTempSegmentFrame, binaryGMMFrame);

	//save into display file
	gmmFrame = gmmTempSegmentFrame;

	//display frame
	displayFrame("GMM Frame", gmmFrame);

	//save mask as main gmmFrame
	gmmFrame = binaryGMMFrame;

	displayFrame("GMM Binary Frame", binaryGMMFrame);

	//if buffer built
	if (i > bufferMemory * 2) {
		//perform sWND
		gmmFrame = slidingWindowNeighborDetector(binaryGMMFrame,
				gmmFrame.rows / 5, gmmFrame.cols / 10);
		displayFrame("sWDNs GMM Frame 1", gmmFrame);

		gmmFrame = slidingWindowNeighborDetector(gmmFrame, gmmFrame.rows / 10,
				gmmFrame.cols / 20);
		displayFrame("sWDNs GMM Frame 2", gmmFrame);

		gmmFrame = slidingWindowNeighborDetector(gmmFrame, gmmFrame.rows / 20,
				gmmFrame.cols / 40);
		displayFrame("sWDNs GMM Frame 3", gmmFrame);

		Mat gmmFrameSWNDCanny = gmmFrame;

		if (i > bufferMemory * 3 - 1) {
			//perform Canny
			gmmFrameSWNDCanny = cannyContourDetector(gmmFrame);
			displayFrame("CannyGMM", gmmFrameSWNDCanny);
		}

		//save into canny
		cannyGMM = gmmFrameSWNDCanny;
	}

	//signal thread completion
	gaussianMixtureModelCompletion = 1;
}

//method to handle GMM thread
Mat gaussianMixtureModel() {

	extern int i;
	extern int bufferMemory;
	extern Mat cannyGMM;
	extern vector<Mat> globalFrames;

	//instantiate thread object
	pthread_t gaussianMixtureModelThread;

	//instantiating multithread Data object
	struct thread_data threadData;

	//save i data
	threadData.data = i;

	//create thread
	pthread_create(&gaussianMixtureModelThread, NULL, calcGaussianMixtureModel,
			(void *) &threadData);

	//return processed frame if completed
	if (i > bufferMemory * 2)
		return cannyGMM;
	//return tmp frame if not finished
	else
		return globalFrames[i];
}



/*
 * generateBackgroundImage.cpp
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

#include "grayScaleFrameMedian.h"

//namespaces for convenience
using namespace cv;
using namespace std;

//method to handle all background image generation
void generateBackgroundImage(int FRAME_RATE) {

	extern bool readMedianImg;
	extern bool useMedians;
	extern int bufferMemory;
	extern int i;
	extern Mat backgroundFrameMedian;
	extern Mat drawAnomalyCar;
	extern Mat backgroundFrameMedianColor;
	extern Mat finalTrackingFrame;
	extern int medianImageCompletion;

	//if post-processing
	if (readMedianImg && useMedians && i < bufferMemory + 5) {
		//read median image
		backgroundFrameMedian = imread("assets/froggerHighwayDrunkMedian.jpg");

		//saving background to image
		backgroundFrameMedian.copyTo(drawAnomalyCar);
		backgroundFrameMedian.copyTo(backgroundFrameMedianColor);

		//convert to grayscale
		cvtColor(backgroundFrameMedian, backgroundFrameMedian, CV_BGR2GRAY);

		displayFrame("backgroundFrameMedian", backgroundFrameMedian);

		//saving background to image
		backgroundFrameMedian.copyTo(finalTrackingFrame);
	}

	//if real-time calculation
	else {
		//after initial buffer read and using medians
		if (i == bufferMemory && useMedians) {
			grayScaleFrameMedian();

			while (medianImageCompletion != 1) {
			}
		}
		//every 3 minutes
		if (i % (FRAME_RATE * 180) == 0 && i > 0) {
			//calculate new medians
			grayScaleFrameMedian();

			while (medianImageCompletion != 1) {
			}

		}
	}
	//signal completion
	medianImageCompletion = 0;
}


/*
 * grayScaleFrameMedian.cpp
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

#include "grayScaleFrameMedian.h"
#include "calcMedian.h"
#include "currentDateTime.h"

//namespaces for convenience
using namespace cv;
using namespace std;

//defining format of data sent to threads
struct thread_data {
	//include int for data passing
	int data;
};

//thread to calculate median of image
void *calcMedianImage(void *threadarg) {
	extern vector <Mat> globalGrayFrames;
	extern int i;
	extern Mat backgroundFrameMedian;
	extern Mat finalTrackingFrame;
	extern Mat drawAnomalyCar;
	extern Mat backgroundFrameMedianColor;
	extern int bufferMemory;
	extern int medianImageCompletion;

	//defining data structure to read in info to new thread
	struct thread_data *data;
	data = (struct thread_data *) threadarg;

	//performing deep copy
	globalGrayFrames[i].copyTo(backgroundFrameMedian);

	//variables to display completion
	double displayPercentageCounter = 0;
	double activeCounter = 0;

	//calculating number of runs
	for (int j = 0; j < backgroundFrameMedian.rows; j++) {
		for (int a = 0; a < backgroundFrameMedian.cols; a++) {
			for (int t = (i - bufferMemory); t < i; t++) {
				displayPercentageCounter++;
			}
		}
	}

	//stepping through all pixels
	for (int j = 0; j < backgroundFrameMedian.rows; j++) {
		for (int a = 0; a < backgroundFrameMedian.cols; a++) {
			//saving all pixel values
			vector<int> pixelHistory;

			//moving through all frames stored in buffer
			for (int t = (i - bufferMemory); t < i; t++) {
				//Mat to store current frame to process
				Mat currentFrameForMedianBackground;

				//copy current frame
				globalGrayFrames.at(i - t).copyTo(
						currentFrameForMedianBackground);

				//save pixel into pixel history
				pixelHistory.push_back(
						currentFrameForMedianBackground.at<uchar>(j, a));

				//increment for load calculations
				activeCounter++;
			}

			//calculate median value and store in background image
			backgroundFrameMedian.at<uchar>(j, a) = calcMedian(pixelHistory);
		}

		//display percentage completed
		cout << ((activeCounter / displayPercentageCounter) * 100)
				<< "% Median Image Scanned" << endl;

	}

	//saving background to write on
	backgroundFrameMedian.copyTo(finalTrackingFrame);
	backgroundFrameMedian.copyTo(drawAnomalyCar);
	backgroundFrameMedian.copyTo(backgroundFrameMedianColor);

	//signal thread completion
	medianImageCompletion = 1;
}


//method to perform median on grayscale images
void grayScaleFrameMedian() {
	extern bool debug;
	extern int i;
	extern Mat backgroundFrameMedian;

	if (debug)
		cout << "Entered gray scale median" << endl;

	//instantiating multithread object
	pthread_t medianImageThread;

	//instantiating multithread Data object
	struct thread_data threadData;

	//saving data into multithread
	threadData.data = i;

	//creating thread to calculate median of image
	pthread_create(&medianImageThread, NULL, calcMedianImage,
			(void *) &threadData);

	//save median image
	imwrite((currentDateTime() + "medianBackgroundImage.jpg"),
			backgroundFrameMedian);
}



/*
 * imageSubtraction.cpp
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
#include "slidingWindowNeighborDetector.h"
#include "cannyContourDetector.h"

#include "vibeBackgroundSubtraction.h"

#include "mogDetection.h"
#include "mogDetection2.h"

#include "medianDetection.h"

#include "grayScaleFrameMedian.h"
#include "calcMedian.h"
#include "currentDateTime.h"
#include "thresholdFrame.h"

//namespaces for convenience
using namespace cv;
using namespace std;

//method to perform simple image subtraction
Mat imageSubtraction() {

	extern vector<Mat> globalGrayFrames;
	extern int i;
	extern Mat backgroundFrameMedian;

	//subtract frames
	Mat tmpStore = globalGrayFrames[i] - backgroundFrameMedian;

	displayFrame("Raw imgSub", tmpStore);
	//threshold frames
	tmpStore = thresholdFrame(tmpStore, 50);
	displayFrame("Thresh imgSub", tmpStore);

	//perform sWND
	tmpStore = slidingWindowNeighborDetector(tmpStore, tmpStore.rows / 5,
			tmpStore.cols / 10);
	displayFrame("SWD", tmpStore);
	tmpStore = slidingWindowNeighborDetector(tmpStore, tmpStore.rows / 10,
			tmpStore.cols / 20);
	displayFrame("SWD2", tmpStore);
	tmpStore = slidingWindowNeighborDetector(tmpStore, tmpStore.rows / 20,
			tmpStore.cols / 40);
	displayFrame("SWD3", tmpStore);

	//perform canny
	tmpStore = cannyContourDetector(tmpStore);
	displayFrame("Canny Contour", tmpStore);

	//return frame
	return tmpStore;
}


/*
 * individualTracking.cpp
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
#include "calculateDeviance.h"

//namespaces for convenience
using namespace cv;
using namespace std;

//method to handle individual tracking
void individualTracking() {
	extern int i;
	extern int bufferMemory;
	extern int mlBuffer;
	extern vector<vector<Point> > carCoordinates;
	extern vector<vector<Point> > vectorOfDetectedCars;
	extern vector<Point> detectedCoordinates;
	extern vector<vector<Point> > coordinateMemory;

	//distance threshold
 	const double distanceThreshold = 25;

 	//bool to show one car is registereds
	bool registerdOnce = false;

	//if ready to begin registering
	if (i == (bufferMemory + mlBuffer + 3) || ((carCoordinates.size() == 0) && i > (bufferMemory + mlBuffer)))
	{
		registerFirstCar();
 	}

 	//if car is in scene
	else if (detectedCoordinates.size() > 0) {

		//save into vector
		vectorOfDetectedCars.push_back(detectedCoordinates);
		coordinateMemory.push_back(detectedCoordinates);

		//calculate deviance of cars
		calculateDeviance();

		//analyze cars movement
		analyzeMovement();
	}
}


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

			}

			//save all vectors into LASM model
			learnedCoordinates.push_back(tmpPointVector);
		}
	}
}




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

			//if LASM changed
			if(sqrt( abs(existingPoint.x - tmpPoint.x) *
					abs(existingPoint.y - tmpPoint.y)) != 0)
			{
				//display difference
				cout << " DIFFERENCE FROM NORMAL " << to_string(sqrt( abs(existingPoint.x - tmpPoint.x) *
					abs(existingPoint.y - tmpPoint.y))) << endl;
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
						int tmpATI = 2;

						//determining average y
						int tmp = ((tmpPoint.y +  oldTmpPoint.y)) / tmpATI;

						//write averaged values to vector
						distanceFromNormal.push_back( abs(	tmpPoint.y - tmpPointVector.at(tmpPoint.x / 7).y));
						distanceFromNormalPoints.push_back(tmpPoint);

						//writing to frame
						putText(distanceFrame, to_string(abs(tmpPoint.y - tmpPointVector.at(tmpPoint.x / 7).y)),
								tmpPoint, 3, 1, Scalar(254, 254, 0), 2);

						//drawing onto frame
						circle(distanceFrame, tmpPoint, 4, Scalar(254, 254, 0), -1, 8, 0);

						//create averaged points
						Point averagePoint(v, tmp);

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
	displayFrame("distanceFrame", distanceFrame, true);
}


/*
 * medianDetection.cpp
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

#include "generateBackgroundImage.h"

#include "imageSubtraction.h"

//namespaces for convenience
using namespace cv;
using namespace std;

//defining format of data sent to threads
struct thread_data {
	//include int for data passing
	int data;
};

//method to handle median image subtraction
Mat medianImageSubtraction(int FRAME_RATE) {
	//generate or read background image
	generateBackgroundImage(FRAME_RATE);

	//calculate image difference and return
	return imageSubtraction();
}

//method to handle median image subtraction
void *computeMedianDetection(void *threadarg) {
	extern Mat medianDetectionGlobalFrame;
	extern int FRAME_RATE;
	extern int medianDetectionGlobalFrameCompletion;

	struct thread_data *data;
	data = (struct thread_data *) threadarg;
	int tmp = data->data;

	medianDetectionGlobalFrame = medianImageSubtraction(FRAME_RATE);

	/*
	 //generate or read background image
	 generateBackgroundImage(FRAME_RATE);

	 //calculate image difference and save to global
	 medianDetectionGlobalFrame = imageSubtraction();
	 */

	medianDetectionGlobalFrameCompletion = 1;
}

void medianDetectionThreadHandler(int FRAME_RATE) {
	//instantiating multithread object
	pthread_t medianDetectionThread;

	//instantiating multithread Data object
	struct thread_data threadData;

	//saving data into data object
	threadData.data = FRAME_RATE;

	//creating threads
	int medianDetectionThreadRC = pthread_create(&medianDetectionThread, NULL,
			computeMedianDetection, (void *) &threadData);
}



/*
 * mogDetection.cpp
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
#include "slidingWindowNeighborDetector.h"
#include "cannyContourDetector.h"

#include "vibeBackgroundSubtraction.h"

//namespaces for convenience
using namespace cv;
using namespace std;

//defining format of data sent to threads
struct thread_data {
	//include int for data passing
	int data;
};


//method to do background subtraction with MOG 1
void *computeBgMog1(void *threadarg) {

	extern BackgroundSubtractorMOG bckSubMOG;
	extern vector <Mat> globalFrames;
	extern int i;
	extern Mat mogDetection1GlobalFrame;
	extern int mogDetection1GlobalFrameCompletion;

	struct thread_data *data;
	data = (struct thread_data *) threadarg;

	//instantiating Mat objects
	Mat fgmask;
	Mat bck;
	Mat fgMaskSWNDCanny;

	//performing background subtraction
	bckSubMOG.operator()(globalFrames.at(i), fgmask, .01); //1.0 / 200);

	displayFrame("MOG Fg MAsk", fgmask);
	displayFrame("RCFrame", globalFrames[i]);

	//performing sWND
	Mat fgmaskSWND = slidingWindowNeighborDetector(fgmask, fgmask.rows / 10,
			fgmask.cols / 20);
	displayFrame("fgmaskSWND", fgmaskSWND);

	fgmaskSWND = slidingWindowNeighborDetector(fgmaskSWND, fgmaskSWND.rows / 20,
			fgmaskSWND.cols / 40);
	displayFrame("fgmaskSWNDSWND2", fgmaskSWND);

	fgmaskSWND = slidingWindowNeighborDetector(fgmaskSWND, fgmaskSWND.rows / 30,
			fgmaskSWND.cols / 60);
	displayFrame("fgmaskSWNDSWND3", fgmaskSWND);

	//performing canny
	fgMaskSWNDCanny = cannyContourDetector(fgmaskSWND);
	displayFrame("fgMaskSWNDCanny2", fgMaskSWNDCanny);

	//return canny
	mogDetection1GlobalFrame = fgMaskSWNDCanny;

	//signal completion
	mogDetection1GlobalFrameCompletion = 1;
}


void mogDetectionThreadHandler(bool buffer) {
	extern int i;

	//instantiating multithread object
	pthread_t mogDetectionThread;

	//instantiating multithread Data object
	struct thread_data threadData;

	//saving data into data object
	threadData.data = i;

	//creating threads
	int mogDetectionThreadRC = pthread_create(&mogDetectionThread, NULL,
			computeBgMog1, (void *) &threadData);
}



/*
 * mogDetection2.cpp
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
#include "slidingWindowNeighborDetector.h"

#include "mogDetection.h"
#include "mogDetection2.h"

#include "cannyContourDetector.h"

//namespaces for convenience
using namespace cv;
using namespace std;

extern int i;

//defining format of data sent to threads
struct thread_data {
	//include int for data passing
	int data;
};

//method to do background subtraction with MOG 2
void *computeBgMog2(void *threadarg) {

	extern vector <Mat> globalFrames;
	extern int i;
	extern Ptr<BackgroundSubtractorMOG2> pMOG2Shadow;
	extern Mat mogDetection2GlobalFrame;
	extern int mogDetection2GlobalFrameCompletion;

	struct thread_data *data;
	data = (struct thread_data *) threadarg;

	//instantiating Mat objects
	Mat fgmaskShadow;
	Mat frameToResizeShadow;

	//copying into tmp variable
	globalFrames[i].copyTo(frameToResizeShadow);

	//performing background subtraction
	pMOG2Shadow->operator()(frameToResizeShadow, fgmaskShadow, .01);

	//performing sWND
	displayFrame("fgmaskShadow", fgmaskShadow);
	Mat fgmaskShadowSWND = slidingWindowNeighborDetector(fgmaskShadow,
			fgmaskShadow.rows / 10, fgmaskShadow.cols / 20);
	displayFrame("fgmaskShadowSWND", fgmaskShadowSWND);

	fgmaskShadowSWND = slidingWindowNeighborDetector(fgmaskShadowSWND,
			fgmaskShadowSWND.rows / 20, fgmaskShadowSWND.cols / 40);
	displayFrame("fgmaskShadowSWND2", fgmaskShadowSWND);

	//performing canny
	Mat fgMaskShadowSWNDCanny = cannyContourDetector(fgmaskShadowSWND);
	displayFrame("fgMaskShadowSWNDCanny2", fgMaskShadowSWNDCanny);

	//return canny
	mogDetection2GlobalFrame = fgMaskShadowSWNDCanny;

	//signal completion
	mogDetection2GlobalFrameCompletion = 1;
}


void mogDetection2ThreadHandler(bool buffer) {
	//instantiating multithread object
	pthread_t mogDetection2Thread;

	//instantiating multithread Data object
	struct thread_data threadData;

	//saving data into data object
	threadData.data = i;

	//creating threads
	int mogDetection2ThreadRC = pthread_create(&mogDetection2Thread, NULL,
			computeBgMog2, (void *) &threadData);
}


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
 			welcome("Median -> Set");
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
 			welcome("ViBe -> Set");

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
 			welcome("MOG1 -> Set");
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
 			welcome("MOG2 -> Set");
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



/*
 * opticalFlowAnalysisObjectDetection.cpp
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

#include "slidingWindowNeighborDetector.h"
#include "cannyContourDetector.h"

//namespaces for convenience
using namespace cv;
using namespace std;

extern vector<Mat> globalFrames;
extern vector<Mat> globalGrayFrames;
extern int i;
extern Mat flow;
extern Mat cflow;
extern bool debug;
extern int opticalFlowAnalysisObjectDetectionThreadCompletion;
extern int opticalFlowThreadCompletion;
extern Mat ofaGlobalHeatMap;
extern Mat thresholdFrameOFA;

//defining format of data sent to threads
struct thread_data {
	//include int for data passing
	int data;
};

//method to perform OFA threshold on Mat
void *computeOpticalFlowAnalysisObjectDetection(void *threadarg) {

	//reading in data sent to thread into local variable
	struct opticalFlowThreadData *data;
	data = (struct opticalFlowThreadData *) threadarg;

	Mat ofaObjectDetection;

	//deep copy grayscale frame
	globalGrayFrames.at(i - 1).copyTo(ofaObjectDetection);

	//set threshold
	const double threshold = 10000;

	//iterating through OFA pixels
	for (int j = 0; j < cflow.rows; j++) {
		for (int a = 0; a < cflow.cols; a++) {
			const Point2f& fxy = flow.at<Point2f>(j, a);

			//if movement is greater than threshold
			if ((sqrt((abs(fxy.x) * abs(fxy.y))) * 10000) > threshold) {
				//write to binary image
				ofaObjectDetection.at<uchar>(j, a) = 255;
			} else {
				//write to binary image
				ofaObjectDetection.at<uchar>(j, a) = 0;
			}
		}
	}

	//performing sWND
	displayFrame("OFAOBJ pre", ofaObjectDetection);

	ofaObjectDetection = slidingWindowNeighborDetector(ofaObjectDetection,
			ofaObjectDetection.rows / 10, ofaObjectDetection.cols / 20);
	displayFrame("sWNDFrame1", ofaObjectDetection);

	ofaObjectDetection = slidingWindowNeighborDetector(ofaObjectDetection,
			ofaObjectDetection.rows / 20, ofaObjectDetection.cols / 40);
	displayFrame("sWNDFrame2", ofaObjectDetection);

	ofaObjectDetection = slidingWindowNeighborDetector(ofaObjectDetection,
			ofaObjectDetection.rows / 30, ofaObjectDetection.cols / 60);
	displayFrame("sWNDFrame3", ofaObjectDetection);

	//saving into heat map
	ofaObjectDetection.copyTo(ofaGlobalHeatMap);

	//running canny detector
	thresholdFrameOFA = cannyContourDetector(ofaObjectDetection);
	displayFrame("sWNDFrameCanny", thresholdFrameOFA);

	//signal thread completion
	opticalFlowAnalysisObjectDetectionThreadCompletion = 1;
}

//method to handle OFA threshold on Mat thread
void opticalFlowAnalysisObjectDetection(Mat& cflowmap, Mat& flow) {
	//instantiating multithread object
	pthread_t opticalFlowAnalysisObjectDetectionThread;

	//instantiating multithread Data object
	struct thread_data threadData;

	//saving data to pass
	threadData.data = i;

	//creating optical flow object thread
	pthread_create(&opticalFlowAnalysisObjectDetectionThread, NULL,
			computeOpticalFlowAnalysisObjectDetection, (void *) &threadData);

}



/*
 * opticalFlowFarneback.cpp
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
#include "opticalFlowAnalysisObjectDetection.h"

#include "blurFrame.h"

//namespaces for convenience
using namespace cv;
using namespace std;

extern vector <Mat> globalFrames;
extern int i;
extern Mat cflow;
extern bool debug;
extern Mat flow;
extern int opticalFlowAnalysisObjectDetectionThreadCompletion;
extern int opticalFlowThreadCompletion;
extern Mat thresholdFrameOFA;

//defining format of data sent to threads
struct thread_data {
	//include int for data passing
	int data;
};

//method to draw optical flow, only should be called during demos
void drawOpticalFlowMap(const Mat& flow, Mat& cflowmap, double,
		const Scalar& color) {
	extern int opticalFlowDensityDisplay;

	//iterating through each pixel and drawing vector
	for (int y = 0; y < cflowmap.rows; y += opticalFlowDensityDisplay) {
		for (int x = 0; x < cflowmap.cols; x += opticalFlowDensityDisplay) {
			const Point2f& fxy = flow.at<Point2f>(y, x);
			line(cflowmap, Point(x, y),
					Point(cvRound(x + fxy.x), cvRound(y + fxy.y)), color);
			circle(cflowmap, Point(x, y), 0, color, -1);
		}
	}
	//display optical flow map
	displayFrame("RFDOFA", cflowmap);
 }

//method to perform optical flow analysis
void *computeOpticalFlowAnalysisThread(void *threadarg) {

	//reading in data sent to thread into local variable
	struct thread_data *data;
	data = (struct thread_data *) threadarg;
	int temp = data->data;

	//defining local variables for FDOFA
	Mat prevFrame, currFrame;
	Mat gray, prevGray;

	//saving images for OFA
	prevFrame = globalFrames[i - 1];
	currFrame = globalFrames[i];

	//blurring frames
	displayFrame("Pre blur", currFrame);
	currFrame = blurFrame("gaussian", currFrame, 15);
	displayFrame("Post blur", currFrame);
	prevFrame = blurFrame("gaussian", prevFrame, 15);

	//converting to grayscale
	cvtColor(currFrame, gray, COLOR_BGR2GRAY);
	cvtColor(prevFrame, prevGray, COLOR_BGR2GRAY);

	//calculating optical flow
	calcOpticalFlowFarneback(prevGray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

	//converting to display format
	cvtColor(prevGray, cflow, COLOR_GRAY2BGR);

	//perform OFA threshold
	opticalFlowAnalysisObjectDetection(flow, cflow);

	//draw optical flow map
	if (debug) {
 		//drawing optical flow vectors
		drawOpticalFlowMap(flow, cflow, 1.5, Scalar(0, 0, 255));
	}

	//wait for completion
	while (opticalFlowAnalysisObjectDetectionThreadCompletion != 1) {
	}

	//wait for completion
	opticalFlowAnalysisObjectDetectionThreadCompletion = 0;

	//signal completion
	opticalFlowThreadCompletion = 1;
}

//method to handle OFA thread
Mat opticalFlowFarneback() {

 	//instantiate thread object
	pthread_t opticalFlowFarneback;

	//instantiating multithread Data object
	struct thread_data threadData;

	//saving data to pass
	threadData.data = i;

	//create OFA thread
	pthread_create(&opticalFlowFarneback, NULL,
			computeOpticalFlowAnalysisThread, (void *) &threadData);

	//waiting for finish
	while (opticalFlowThreadCompletion != 1) {
	}

	//resetting completion variable
	opticalFlowThreadCompletion = 0;

	//return OFA frame
	return thresholdFrameOFA;
}



/*
 * pollOFAData.cpp
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

//method to poll OFA map
void pollOFAData() {
	extern vector <Point> detectedCoordinates;
	extern Mat ofaGlobalHeatMap;

	//cycle through all detected coordinates
	for (int v = 0; v < detectedCoordinates.size(); v++) {
		//save tmp point
		Point tmpPoint = detectedCoordinates[v];

		//output OFA value
		cout << "OFA VALUE"
				<< ((double) ofaGlobalHeatMap.at<uchar>(tmpPoint.x, tmpPoint.y))
				<< endl;
	}
}


/*
 * processCoordinates.cpp
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
#include "displayCoordinates.h"
#include "drawCoordinates.h"

//method to handle coordinates
void processCoordinates() {

	extern String fileTime;
	extern Mat finalTrackingFrame;
	extern vector<Point> detectedCoordinates;
	extern int numberOfCars;

	const int averageThreshold = 85;

	//draw raw coordinates
	drawCoordinates(detectedCoordinates, "1st Pass");

	//write to file
	imwrite(fileTime + "finalTrackingFrame.TIFF", finalTrackingFrame);

	//average points using threshold
	detectedCoordinates = averageCoordinates(detectedCoordinates, averageThreshold);

	//count number of cars
	numberOfCars = detectedCoordinates.size();

	//draw processed coordinates
	drawCoordinates(detectedCoordinates, "2nd Pass");
}


/*
 * processExit.cpp
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

#include "computeRunTime.h"

//namespaces for convenience
using namespace cv;
using namespace std;

//method to process exit of software
bool processExit(VideoCapture capture, clock_t t1, char keyboardClick)
{
	extern vector<Mat> globalFrames;

	//if escape key is pressed
	if (keyboardClick == 27)
	{
		//display exiting message
		cout << "Exiting" << endl;

		//compute total run time
		computeRunTime(t1, clock(), (int) capture.get(CV_CAP_PROP_POS_FRAMES));

		//delete entire vector
		globalFrames.erase(globalFrames.begin(), globalFrames.end());

		//report file finished writing
		cout << "Finished writing file, Goodbye." << endl;

		//exit program
		return true;
	}

	else
	{
		return false;
	}
}



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

	//save all cars
	vectorOfDetectedCars.push_back(detectedCoordinates);

	//cycling through points
	for (int v = 0; v < detectedCoordinates.size(); v++) {

		//if in the starting area on either side
		if (detectedCoordinates[v].x < 75 || detectedCoordinates[v].x >FRAME_WIDTH - 75) {
			//creating vector of car coordinates
			vector<Point> carCoordinate;

			//saving car coordinates
			carCoordinate.push_back(detectedCoordinates[v]);
			carCoordinates.push_back(carCoordinate);
		}
	}
}




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



/*
 * sortCoordinates.cpp
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

//namespaces for convenience
using namespace cv;
using namespace std;

//method to compare points
bool point_comparator(const Point2f &a, const Point2f &b) {
	//determining difference between distances of points
	return a.x * a.x + a.y * a.y < b.x * b.x + b.y * b.y;
}

//method to sort all coordinates
vector<Point> sortCoordinates(vector<Point> coordinates) {
	//sort using point_compartor
	sort(coordinates.begin(), coordinates.end(), point_comparator);

	//return sorted coordinates
	return coordinates;
}


/*
 * thresholdFrame.cpp
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

//namespaces for convenience
using namespace cv;
using namespace std;


//method to threshold standard frame
Mat thresholdFrame(Mat sourceDiffFrame, const int threshold) {
	//Mat to hold frame
	Mat thresholdFrame;

	//perform deep copy into destination Mat
	sourceDiffFrame.copyTo(thresholdFrame);

	//steping through pixels
	for (int j = 0; j < sourceDiffFrame.rows; j++) {
		for (int a = 0; a < sourceDiffFrame.cols; a++) {
			//if pixel value greater than threshold
			if (sourceDiffFrame.at<uchar>(j, a) > threshold) {
				//write to binary image
				thresholdFrame.at<uchar>(j, a) = 255;
			} else {
				//write to binary image
				thresholdFrame.at<uchar>(j, a) = 0;
			}
		}
	}

	//return thresholded frame
	return thresholdFrame;
}


/*
 * trackingML.cpp
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
#include "drawTmpTracking.h"
#include "drawAllTracking.h"

//namespaces for convenience
using namespace cv;
using namespace std;

//method to handle all tracking Machine Learning commands
void trackingML()
{
	extern int i;
	extern int bufferMemory;
	extern int mlBuffer;
	extern vector<Point> detectedCoordinates;

	//if CV is still initializing
	if (i <= bufferMemory + mlBuffer) {
		//display welcome image
		welcome(
				"Final Initialization; Running ML Startup -> Frames Remaining: "
						+ to_string((bufferMemory + mlBuffer + 1) - i));
	}

	//if ready to run
	else if (i > bufferMemory + mlBuffer + 1) {

		//if booting ML
		if (i == bufferMemory + mlBuffer + 1) {
			//display bootup message
			welcome("Initialization Complete -> Starting ML");
		}

		//begin processing frames
		else if (i > bufferMemory + mlBuffer + 1) {
			//display ready to run
			String tmpToDisplay = "Running ML Tracking -> Frame Number: "
					+ to_string(i);

			//display welcome image
			welcome(tmpToDisplay);
		}

		//process coordinates and average
		processCoordinates();

		//tracking all individual cars
		individualTracking();

		//draw coordinates in the tmp
		drawTmpTracking();

		//draw all car points
		drawAllTracking();
	}

	//erase detected coordinates for next run
	detectedCoordinates.erase(detectedCoordinates.begin(), detectedCoordinates.end());
}


/*
 * type2StrTest.cpp
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

//namespaces for convenience
using namespace cv;
using namespace std;

//method to identify type of Mat based on identifier
string type2str(int type) {

	//string to return type of mat
	string r;

	//stats about frame
	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	//switch to determine Mat type
	switch (depth) {
		case CV_8U:
			r = "8U";
			break;
		case CV_8S:
			r = "8S";
			break;
		case CV_16U:
			r = "16U";
			break;
		case CV_16S:
			r = "16S";
			break;
		case CV_32S:
			r = "32S";
			break;
		case CV_32F:
			r = "32F";
			break;
		case CV_64F:
			r = "64F";
			break;
		default:
			r = "User";
			break;
	}

	//append formatting
	r += "C";
	r += (chans + '0');

	//return Mat type
	return r;
}


/*
 * vibeBackgroundSubtraction.cpp
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
#include "slidingWindowNeighborDetector.h"
#include "cannyContourDetector.h"

//namespaces for convenience
using namespace cv;
using namespace std;

extern int i;
extern int bufferMemory;
extern vector <Mat> globalFrames;
extern Mat resizedFrame;
extern Mat vibeDetectionGlobalFrame;
extern int vibeDetectionGlobalFrameCompletion;
extern bgfg_vibe bgfg;
extern Mat vibeBckFrame;

//defining format of data sent to threads
struct thread_data {
	//include int for data passing
	int data;
};

//method to perform vibe background subtraction
void *computeVibeBackgroundThread(void *threadarg) {
 	struct thread_data *data;
	data = (struct thread_data *) threadarg;

	//instantiating Mat frame object
	Mat sWNDVibeCanny;

	//if done buffering
	if (i == bufferMemory) {
		//instantiating Mat frame object
		Mat resizedFrame;

		//saving current frame
		globalFrames[i].copyTo(resizedFrame);

		//initializing model
		bgfg.init_model(resizedFrame);

		//return tmp frame
		vibeDetectionGlobalFrame = sWNDVibeCanny;

		vibeDetectionGlobalFrameCompletion = 1;
	}

	else {
		//instantiating Mat frame object
		Mat resizedFrame;

		//saving current frame
		globalFrames[i].copyTo(resizedFrame);

		//processing model
		vibeBckFrame = *bgfg.fg(resizedFrame);

		displayFrame("vibeBckFrame", vibeBckFrame);

		//performing sWND
		Mat sWNDVibe = slidingWindowNeighborDetector(vibeBckFrame,
				vibeBckFrame.rows / 10, vibeBckFrame.cols / 20);
		displayFrame("sWNDVibe1", sWNDVibe);

		//performing sWND
		sWNDVibe = slidingWindowNeighborDetector(vibeBckFrame,
				vibeBckFrame.rows / 20, vibeBckFrame.cols / 40);
		displayFrame("sWNDVibe2", sWNDVibe);

		Mat sWNDVibeCanny = sWNDVibe;

		if (i > bufferMemory * 3 - 1) {
			//performing canny
			Mat sWNDVibeCanny = cannyContourDetector(sWNDVibe);
			displayFrame("sWNDVibeCannycanny2", sWNDVibeCanny);
		}

		//saving processed frame
		vibeDetectionGlobalFrame = sWNDVibeCanny;

		//signalling completion
		vibeDetectionGlobalFrameCompletion = 1;
	}
}

void vibeBackgroundSubtractionThreadHandler(bool buffer) {
	//instantiating multithread object
	pthread_t vibeBackgroundSubtractionThread;

	//instantiating multithread Data object
	struct thread_data threadData;

	//saving data into data object
	threadData.data = i;

	//creating threads
	int vibeBackgroundThreadRC = pthread_create(
			&vibeBackgroundSubtractionThread, NULL, computeVibeBackgroundThread,
			(void *) &threadData);
}

