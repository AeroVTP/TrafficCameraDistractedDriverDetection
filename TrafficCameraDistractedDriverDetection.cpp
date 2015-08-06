//======================================================================================================
// Name        : TrafficCameraDistractedDriverDetection.cpp
// Author      : Vidur Prasad
// Version     : 0.6.6
// Copyright   : Institute for the Development and Commercialization of Advanced Sensor Technology Inc.
// Description : Detect Drunk, Distracted, and Anomalous Driving Using Traffic Cameras
//======================================================================================================

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

////global variables////
//multithreading global variables
vector<Mat> globalFrames;
vector<Mat> globalGrayFrames;

vector<Point> detectedCoordinates;

CvHaarClassifierCascade *cascade;
CvMemStorage *storage;

//vibe constructors
bgfg_vibe bgfg;
BackgroundSubtractorMOG bckSubMOG; // (200,  1, .7, 15);

//global frame properties
int FRAME_HEIGHT;
int FRAME_WIDTH;
int FRAME_RATE;

//x and y car entry
int xLimiter = 40;
int yLimiter = 30;
int xFarLimiter = 60;

int detectStrength = 0;

//global counter
int i = 0;

//global completion variables for multithreading
int medianImageCompletion = 0;
int medianColorImageCompletion = 0;
int opticalFlowThreadCompletion = 0;
int opticalFlowAnalysisObjectDetectionThreadCompletion = 0;
int gaussianMixtureModelCompletion = 0;
int vibeDetectionGlobalFrameCompletion = 0;
int mogDetection1GlobalFrameCompletion = 0;
int mogDetection2GlobalFrameCompletion = 0;
int medianDetectionGlobalFrameCompletion = 0;

double learnedLASMDistance = 0;
double learnedLASMDistanceSum = 0;
double learnedLASMDistanceAccess = 0;

//background subtraction models
Ptr<BackgroundSubtractorGMG> backgroundSubtractorGMM = Algorithm::create<
		BackgroundSubtractorGMG>("BackgroundSubtractor.GMG");
Ptr<BackgroundSubtractorMOG2> pMOG2 = new BackgroundSubtractorMOG2(500, 64,
		true);
Ptr<BackgroundSubtractorMOG2> pMOG2Shadow = new BackgroundSubtractorMOG2(500,
		64, false);

//matrix holding temporary frame after threshold
Mat thresholdFrameOFA;

//matrix to hold GMM frame
Mat gmmFrame;

//matrix holding vibe frame
Mat vibeBckFrame;
Mat vibeDetectionGlobalFrame;

//matrix storing GMM canny
Mat cannyGMM;

//matrix storing OFA thresh operations
Mat ofaThreshFrame;

//Mat to hold Mog1 frame
Mat mogDetection1GlobalFrame;

//Mat to hold Mog2 frame
Mat mogDetection2GlobalFrame;

//Mat objects to hold background frames
Mat backgroundFrameMedian;
Mat backgroundFrameMedianColor;
Mat medianDetectionGlobalFrame;

//Mat for color background frame
Mat backgroundFrameColorMedian;

//Mat to hold temp GMM models
Mat gmmFrameRaw, binaryGMMFrame, gmmTempSegmentFrame;

Mat finalTrackingFrame;
Mat drawAnomalyCar;

//Mat for optical flow
Mat flow;
Mat cflow;
Mat optFlow;

//Mat for OFA Heat Map
Mat ofaGlobalHeatMap;

//vector of vector points for detects
vector<vector<Point> > carCoordinates;
vector<vector<Point> > vectorOfDetectedCars;

//current frame detected coordinates
vector<Point> globalDetectedCoordinates;

//if first time performing OFA
bool objectOFAFirstTime = true;

//optical flow density
int opticalFlowDensityDisplay = 5;

//Buffer memory size
int bufferMemory = 90;

//boolean to decide if preprocessed median should be used
bool readMedianImg = true;

//controls all displayFrame statements
bool debug = false;

//controls if median is used
bool useMedians = true;

//variables for learned models
double xAverageMovement = 0;
double xAverageCounter = 0;
double xLearnedMovement = 0;

double yAverageMovement = 0;
double yAverageCounter = 0;
double yLearnedMovement = 0;

double learnedAggregate = 0;
double learnedAngle = 0;

double currentSpeed = 0;
double learnedSpeed = 0;
double learnedSpeedAverage = 0;

double currentDistance = 0;
double learnedDistance = 0;
double learnedDistanceAverage = 0;
double learnedDistanceCounter = 0;

//buffer memory for ML
 int mlBuffer = 3;

//number of Cars in frame
int numberOfCars = 0;

int lastAnomalyDetectedFN = 0;
int numberOfAnomaliesDetected = 0;

//string to hold start time
String fileTime;

//vectors for learning
vector<int> lanePositions;
vector<vector<Point> > coordinateMemory;
vector<vector<Point> > learnedCoordinates;
vector<vector<Point> > accessTimes;
vector<vector<int> > accessTimesInt;
vector<double> distanceFromNormal;
vector<Point> distanceFromNormalPoints;

//setting constant filename to read form
//const char* filename = "assets/testRecordingSystemTCD3TCheck.mp4";
//const char* filename = "assets/ElginHighWayTCheck.mp4";
//const char* filename = "assets/SCDOTTestFootageTCheck.mov";
//const char* filename = "assets/sussexGardensPaddingtonLondonShortElongtedTCheck.mp4";
//const char* filename = "assets/sussexGardenPaddingtonLondonFullTCheck.mp4";
//const char* filename = "assets/sussexGardenPaddingtonLondonFullEFPSTCheck.mp4";
//const char* filename = "assets/sussexGardenPaddingtonLondonFPS15TCheck.mp4";
//const char* filename = "assets/sussexGardenPaddingtonLondonFPS10TCheck.mp4";
//const char* filename = "assets/genericHighWayYoutubeStockFootage720OrangeHDCom.mp4";
//const char* filename = "assets/xmlTrainingVideoSet2.mp4";
//const char* filename = "assets/videoAndrewGithub.mp4";
//const char* filename = "assets/froggerHighwayTCheck.mp4";
//const char* filename = "assets/OrangeHDStockFootageHighway72010FPSTCheck.mp4";
//const char* filename = "assets/trafficStockCountryRoad16Seconds30FPSTCheck.mp4";
//const char* filename = "assets/cityCarsTraffic3LanesStreetRoadJunctionPond5TCheck15FPSTCheck.mp4";
//const char* filename = "assets/froggerHighwayDrunk.mp4";
//const char* filename = "assets/froggerHighwayDrunkShort.mp4";
//const char* filename = "assets/froggerHighwayDrunkV2.mp4";
//const char* filename = "assets/froggerHighwayDrunkV4TCheck.mp4";
//const char* filename = "assets/froggerHighwayLaneChangeV4.mp4";
//const char* filename = "assets/froggerHighwayTCheckV5.mp4";
const char* filename = "assets/froggerHighwayLaneChangeV6.mp4";

//String medianImageFilename = "smallFroggerMedian.jpg";
String medianImageFilename = "froggerHighwayDrunkMedian.jpg";

//defining format of data sent to threads 
struct thread_data {
	//include int for data passing
	int data;
};

//main method
int main() {

	//display welcome message if production code
	if (!debug)
		welcome();

	//creating initial and final clock objects
	//taking current time when run starts
	clock_t t1 = clock();

	//random number generator
	RNG rng(12345);

	//defining VideoCapture object and filename to capture from
	VideoCapture capture(filename);

	//collecting statistics about the video
	//constants that will not change
	const int NUMBER_OF_FRAMES = (int) capture.get(CV_CAP_PROP_FRAME_COUNT);
	FRAME_RATE = (int) capture.get(CV_CAP_PROP_FPS);
	FRAME_WIDTH = capture.get(CV_CAP_PROP_FRAME_WIDTH);
	FRAME_HEIGHT = capture.get(CV_CAP_PROP_FRAME_HEIGHT);

	writeInitialStats(NUMBER_OF_FRAMES, FRAME_RATE, FRAME_WIDTH, FRAME_HEIGHT,
			filename);

	// declaring and initially setting variables that will be actively updated during runtime
	int framesRead = (int) capture.get(CV_CAP_PROP_POS_FRAMES);
	double framesTimeLeft = (capture.get(CV_CAP_PROP_POS_MSEC)) / 1000;

	//creating placeholder object
	Mat placeHolder = Mat::eye(1, 1, CV_64F);

	//vector to store execution times
	vector<string> FPS;

	//string to display execution time
	string strActiveTimeDifference;

	//actual run time, while video is not finished
	while (framesRead < NUMBER_OF_FRAMES) {
		clock_t tStart = clock();

		//read in current key press
		char keyboardClick = cvWaitKey(33);

		//create pointer to new object
		Mat * frameToBeDisplayed = new Mat();

		//creating pointer to new object
		Mat * tmpGrayScale = new Mat();

		//reading in current frame
		capture.read(*frameToBeDisplayed);

		//for initial buffer read
		while (i < bufferMemory) {
			//create pointer to new object
			Mat * frameToBeDisplayed = new Mat();

			//creating pointer to new object
			Mat * tmpGrayScale = new Mat();

			//reading in current frame
			capture.read(*frameToBeDisplayed);

			//adding current frame to vector/array list of matricies
			globalFrames.push_back(*frameToBeDisplayed);

			//convert to gray scale frame
			cvtColor(globalFrames[i], *tmpGrayScale, CV_BGR2GRAY);

			//save grayscale frame
			globalGrayFrames.push_back(*tmpGrayScale);

			//initilize Mat objects
			initilizeMat();

			//display buffer progress
			if (!debug)
				cout << "Buffering frame " << i << ", " << (bufferMemory - i)
						<< " frames remaining." << endl;

			//display splash screen
			welcome();

			//incrementing global counter
			i++;
		}

		//adding current frame to vector/array list of matricies
		globalFrames.push_back(*frameToBeDisplayed);
		Mat dispFrame;
		globalFrames[i].copyTo(dispFrame);
		putText(dispFrame, to_string(i), Point(0, 50), 3, 1, Scalar(0, 255, 0),
				2);

		//display raw frame
		displayFrame("RCFrame", globalFrames[i]);

		//convert to gray scale
		cvtColor(globalFrames[i], *tmpGrayScale, CV_BGR2GRAY);

		//save gray scale frames
		globalGrayFrames.push_back(*tmpGrayScale);

		//gather real time statistics
		framesRead = (int) capture.get(CV_CAP_PROP_POS_FRAMES);
		framesTimeLeft = (capture.get(CV_CAP_PROP_POS_MSEC)) / 1000;

		//clocking end of run time
		clock_t tFinal = clock();

		//calculate time
		strActiveTimeDifference =
				(to_string(calculateFPS(tStart, tFinal))).substr(0, 4);

		//display performance
		if (debug)
			cout << "FPS is "
					<< (to_string(1 / (calculateFPS(tStart, tFinal)))).substr(0,
							4) << endl;

		//saving FPS values
		FPS.push_back(strActiveTimeDifference);

		welcome("Running Computer Vision -> FN:  " + to_string(i));

		//running Computer Vision
		objectDetection(FRAME_RATE);

		welcome("Starting Tracking ML -> FN: " + to_string(i));

		//running Tracking ML
		trackingML();

		welcome("Finished Tracking ML -> FN: " + to_string(i));

		//display frame number
		cout << "Currently processing frame number " << i << "." << endl;

		//method to process exit
		if(processExit(capture,  t1, keyboardClick))
			return 0;

		//deleting current frame from RAM
		delete frameToBeDisplayed;

		//incrementing global counter
		i++;
	}

	//delete entire vector
	globalFrames.erase(globalFrames.begin(), globalFrames.end());

	//compute run time
	computeRunTime(t1, clock(), (int) capture.get(CV_CAP_PROP_POS_FRAMES));

	//display finished, promt to close program
	cout << "Execution finished, file written, click to close window. " << endl;

	//wait for button press to proceed
	waitKey(0);

	//return code is finished and ran successfully
	return 0;
}

