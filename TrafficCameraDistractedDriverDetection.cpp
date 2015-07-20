//======================================================================================================
// Name        : TrafficCameraDistractedDriverDetection.cpp
// Author      : Vidur Prasad
// Version     : 0.4.0
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

//include CMT
#include "CMT.h"

//include CCV
#include <ccv.h>

//namespaces for convenience
using namespace cv;
using namespace std;

////global variables////

//multithreading global variables
vector <Mat> globalFrames;
vector <Mat> globalGrayFrames;

vector <Point> detectedCoordinates;

CvHaarClassifierCascade *cascade;
CvMemStorage  *storage;

//vibe constructors
bgfg_vibe bgfg;
BackgroundSubtractorMOG bckSubMOG; // (200,  1, .7, 15);
 
//global frame properties
int FRAME_HEIGHT;
int FRAME_WIDTH;
int FRAME_RATE;

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

//background subtraction models
Ptr<BackgroundSubtractorGMG> backgroundSubtractorGMM = Algorithm::create<BackgroundSubtractorGMG>("BackgroundSubtractor.GMG");
Ptr<BackgroundSubtractorMOG2> pMOG2 = new BackgroundSubtractorMOG2(500, 64, true);
Ptr<BackgroundSubtractorMOG2> pMOG2Shadow = new BackgroundSubtractorMOG2(500, 64, false);
 
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
Mat medianDetectionGlobalFrame;

//Mat for color background frame
Mat backgroundFrameColorMedian;
 
//Mat to hold temp GMM models
Mat gmmFrameRaw, binaryGMMFrame, gmmTempSegmentFrame;

Mat finalTrackingFrame;

//Mat for optical flow
Mat flow;
Mat cflow;
Mat optFlow;

vector <vector <Point> > carCoordinates;

vector <vector <Point> >  vectorOfDetectedCars;

bool objectOFAFirstTime = true;

//optical flow density
int opticalFlowDensityDisplay = 5;

//Buffer memory size
const int bufferMemory = 50;

//boolean to decide if preprocessed median should be used
bool readMedianImg = true;

//controls all displayFrame statements
bool debug = false;

//controls if median is used
bool useMedians = true;

double xAverageMovement = 0;
double xAverageCounter = 0;
double xLearnedMovement = 0;

double yAverageMovement = 0;
double yAverageCounter = 0;
double yLearnedMovement = 0;

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
const char* filename = "assets/froggerHighwayDrunkShort.mp4";

//defining format of data sent to threads 
struct thread_data{
   //include int for data passing
   int data;
};	

//function prototypes
Mat slidingWindowNeighborDetector(Mat srcFrame, int numRowSections, int numColumnSections);
Mat cannyContourDetector(Mat srcFrame);
void fillCoordinates(vector <Point2f> detectedCoordinates);

//method to display frame
void displayFrame(string filename, Mat matToDisplay)
{
	//if in debug mode and Mat is not empty
	if(debug && matToDisplay.size[0] != 0)
	{imshow(filename, matToDisplay);}

	else if(matToDisplay.size[0] == 0)
	{
		cout << filename << " is empty, cannot be displayed." << endl;
	}
}

//method to display frame overriding debug
void displayFrame(string filename, Mat matToDisplay, bool override)
{
	//if override and Mat is not emptys
	if(override && matToDisplay.size[0] != 0 && filename != "Welcome"){ imshow(filename, matToDisplay);}
	else if(override && matToDisplay.size[0] != 0){namedWindow(filename); imshow(filename, matToDisplay);}
	else if(matToDisplay.size[0] == 0)
	{
		cout << filename << " is empty, cannot be displayed." << endl;
	}
}

//method to draw optical flow, only should be called during demos
static void drawOptFlowMap(const Mat& flow, Mat& cflowmap,
                    double, const Scalar& color)
{
	//iterating through each pixel and drawing vector
    for(int y = 0; y < cflowmap.rows; y += opticalFlowDensityDisplay)
    {
    	for(int x = 0; x < cflowmap.cols; x += opticalFlowDensityDisplay)
        {
            const Point2f& fxy = flow.at<Point2f>(y, x);
            line(cflowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),
                 color);
            circle(cflowmap, Point(x,y), 0, color, -1);
        }
   	}
    //display optical flow map
    displayFrame("RFDOFA", cflowmap);
    imshow("RFDOFA", cflowmap);
}

//method that returns date and time as a string to tag txt files
const string currentDateTime()
{
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

    //returning the string with the time
    return buf;
}

//method to apply morphology
Mat morph(Mat sourceFrame, int amplitude, string type)
{
	//using default values
	double morph_size = .5;

	//performing two iterations
	const int iterations = 2;

	//constructing manipulation Mat
	Mat element = getStructuringElement(MORPH_RECT, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );

	//if performing morphological closing
	if(type == "closing")
	{
		//repeat for increased effect
	    for(int v = 0; v < amplitude; v++)
		{
			 morphologyEx(sourceFrame, sourceFrame, MORPH_CLOSE, element,
					 Point(-1,-1), iterations, BORDER_CONSTANT, morphologyDefaultBorderValue());
		}
	}

	//if performing morphological opening
	else if(type == "opening")
	{
		for(int v = 0; v < amplitude; v++)
		{
			//repeat for increased effect
			morphologyEx(sourceFrame, sourceFrame, MORPH_OPEN, element,
					Point(-1,-1), iterations, BORDER_CONSTANT, morphologyDefaultBorderValue());

		}
	}

	else if(type == "erode")
	{
		erode(sourceFrame, sourceFrame,  element);
	}

	//if performing morphological gradient
	else if(type == "gradient")
	{
		//repeat for increased effect
		for(int v = 0; v < amplitude; v++)
		{
			 morphologyEx(sourceFrame, sourceFrame, MORPH_GRADIENT, element,
					 Point(-1,-1), iterations, BORDER_CONSTANT, morphologyDefaultBorderValue());
		}
	}

	//if performing morphological tophat
	else if(type == "tophat")
	{
		//repeat for increased effect
		for(int v = 0; v < amplitude; v++)
		{
			 morphologyEx(sourceFrame, sourceFrame, MORPH_TOPHAT, element,
					 Point(-1,-1), iterations, BORDER_CONSTANT, morphologyDefaultBorderValue());
		}
	}

	//if performing morphological blackhat
	else if(type == "blackhat")
	{
		//repeat for increased effect
		for(int v = 0; v < amplitude; v++)
		{
		morphologyEx(sourceFrame, sourceFrame, MORPH_BLACKHAT, element,
							 Point(-1,-1), iterations, BORDER_CONSTANT, morphologyDefaultBorderValue());
		}
	}

	//if current morph operation is not availble
	else
	{
		//report cannot be done
		if(debug)
			cout << type <<  " type of morphology not implemented yet" << endl;
	}

	//return edited frame
    return sourceFrame;
}

//method to calculate center point of contour
vector <int> centerPoint(vector <Point> contours)
{
	//initializing coordinate variables
	int xTotal = 0;
	int yTotal = 0;

	//vector to hold center point
	vector <int> centerPoint;

	//iterating through each contour
	for(int v = 0 ; v < contours.size() ; v++)
	{
		Point p = contours.at(v);

		xTotal += p.x;
		yTotal += p.y;
	}

	//averaging points
	centerPoint.push_back(xTotal / (contours.size()/2));
	centerPoint.push_back(yTotal / (contours.size()/2));

	//return both center points
	return centerPoint;
}

//method to blur Mat using custom kernel size
Mat blurFrame(string blurType, Mat sourceDiffFrame, int blurSize)
{
	//Mat to hold blurred frame
	Mat blurredFrame;

	//if gaussian blur
	if(blurType == "gaussian")
	{
		//blur frame using custom kernel size
		blur(sourceDiffFrame, blurredFrame, Size (blurSize,blurSize), Point(-1,-1));

		//display blurred frame
		//displayFrame("Gauss Frame", blurredFrame);

		//return blurred frame
		return blurredFrame;
	}

	//if blur type not implemented
	else
	{
		//report not implemented
		if(debug)
			cout << blurType <<  " type of blur not implemented yet" << endl;

		//return original frame
		return sourceDiffFrame;
	}

}

//method to perform OFA threshold on Mat
void *computeOpticalFlowAnalysisObjectDetection(void *threadarg)
{
	//reading in data sent to thread into local variable
	struct opticalFlowThreadData *data;
	data = (struct opticalFlowThreadData *) threadarg;

	Mat ofaObjectDetection; 

	//deep copy grayscale frame
	globalGrayFrames.at(i-1).copyTo(ofaObjectDetection);

	//set threshold
	const double threshold = 10000;

	//iterating through OFA pixels
	for(int j = 0; j < cflow.rows; j++)
	{
		for (int a = 0 ; a < cflow.cols; a++)
		{
			const Point2f& fxy = flow.at<Point2f>(j, a);

 			//if movement is greater than threshold
			if((sqrt((abs(fxy.x) * abs(fxy.y))) * 10000) > threshold)
			{
				//write to binary image
				ofaObjectDetection.at<uchar>(j,a) = 255;
			}
			else
			{
				//write to binary image
				ofaObjectDetection.at<uchar>(j,a) = 0;
			}
		}
	}

	//performing sWND
	displayFrame("OFAOBJ pre" , ofaObjectDetection );
	ofaObjectDetection = slidingWindowNeighborDetector(ofaObjectDetection, ofaObjectDetection.rows / 10, ofaObjectDetection.cols / 20);
	displayFrame("sWNDFrame1" , ofaObjectDetection );
	ofaObjectDetection = slidingWindowNeighborDetector(ofaObjectDetection, ofaObjectDetection.rows / 20, ofaObjectDetection.cols / 40);
	displayFrame("sWNDFrame2" , ofaObjectDetection );
	ofaObjectDetection = slidingWindowNeighborDetector(ofaObjectDetection, ofaObjectDetection.rows / 30, ofaObjectDetection.cols / 60);
	displayFrame("sWNDFrame3" , ofaObjectDetection );
	thresholdFrameOFA = cannyContourDetector(ofaObjectDetection);
	displayFrame("sWNDFrameCanny" , thresholdFrameOFA); 

	//signal thread completion
   	opticalFlowAnalysisObjectDetectionThreadCompletion = 1;
}

//method to handle OFA threshold on Mat thread
void opticalFlowAnalysisObjectDetection(Mat& cflowmap, Mat& flow)
{
	//instantiating multithread object
	pthread_t opticalFlowAnalysisObjectDetectionThread;

	//instantiating multithread Data object
	struct thread_data threadData;

	//saving data to pass
	threadData.data = i;

	//creating optical flow object thread
	pthread_create(&opticalFlowAnalysisObjectDetectionThread, NULL, computeOpticalFlowAnalysisObjectDetection, (void *)&threadData);

}

//method to perform optical flow analysis
void *computeOpticalFlowAnalysisThread(void *threadarg)
{
	//reading in data sent to thread into local variable
	struct thread_data *data;
	data = (struct thread_data *) threadarg;
	int temp = data->data;

	//defining local variables for FDOFA
	Mat prevFrame, currFrame;
	Mat gray, prevGray;

	if(i > 5)
	{
		prevFrame = globalFrames[i-1];
		currFrame = globalFrames[i];
	}

	else
	{
		prevFrame = globalFrames[i];
		currFrame = globalFrames[i];
	}

	displayFrame("Pre blur", currFrame);
	currFrame = blurFrame("gaussian" , currFrame, 15);
	displayFrame("Post blur", currFrame);
	prevFrame = blurFrame("gaussian", prevFrame, 15);

	//converting to grayscale
	cvtColor(currFrame, gray,COLOR_BGR2GRAY);
	cvtColor(prevFrame, prevGray, COLOR_BGR2GRAY);

	//calculating optical flow
	calcOpticalFlowFarneback(prevGray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

	//converting to display format
	cvtColor(prevGray, cflow, COLOR_GRAY2BGR);
 
	//perform OFA threshold
    opticalFlowAnalysisObjectDetection(flow, cflow);

    //draw optical flow map
	if(debug)
	{
		//drawing optical flow vectors
		drawOptFlowMap(flow, cflow, 1.5, Scalar(0, 0, 255));
	}

	//wait for completion
	while(opticalFlowAnalysisObjectDetectionThreadCompletion != 1){}

	//wait for completion
	opticalFlowAnalysisObjectDetectionThreadCompletion = 0;

	opticalFlowThreadCompletion = 1;	
}


//method to do background subtraction with MOG2 
Mat bgMog2(bool buffer)
{

	//instantiating Mat objects
	Mat fgmaskShadow;
	Mat frameToResizeShadow;
	
 	//copying into tmp variable
 	globalFrames[i].copyTo(frameToResizeShadow);
 
 	//performing background subtraction 
	pMOG2Shadow->operator()(frameToResizeShadow , fgmaskShadow, .01);

	//performing sWND
	displayFrame("fgmaskShadow" , fgmaskShadow);
	Mat fgmaskShadowSWND = slidingWindowNeighborDetector(fgmaskShadow, fgmaskShadow.rows/10, fgmaskShadow.cols/20);
	displayFrame("fgmaskShadowSWND", fgmaskShadowSWND); 

	fgmaskShadowSWND = slidingWindowNeighborDetector(fgmaskShadowSWND, fgmaskShadowSWND.rows/20, fgmaskShadowSWND.cols/40);
	displayFrame("fgmaskShadowSWND2", fgmaskShadowSWND);

	//performing canny
	Mat fgMaskShadowSWNDCanny = cannyContourDetector(fgmaskShadowSWND);
	displayFrame("fgMaskShadowSWNDCanny2", fgMaskShadowSWNDCanny);

	//returning processed frame
	return fgMaskShadowSWNDCanny;
}

//method to perform vibe background subtraction
Mat vibeBackgroundSubtraction(bool buffer)
{
	//instantiating Mat frame object
	Mat sWNDVibeCanny;

	//if done buffering
	if(i == bufferMemory)
	{
		//instantiating Mat frame object
		Mat resizedFrame;
		
		//saving current frame
		globalFrames[i].copyTo(resizedFrame);
 
 		//initializing model
 		bgfg.init_model(resizedFrame);

 		//return tmp frame
		return resizedFrame;
	}

	else
	{

		//instantiating Mat frame object
		Mat resizedFrame;

		//saving current frame
		globalFrames[i].copyTo(resizedFrame); 

		//processing model
		vibeBckFrame = *bgfg.fg(resizedFrame);

		displayFrame("vibeBckFrame", vibeBckFrame); 

		//performing sWND
		Mat sWNDVibe = slidingWindowNeighborDetector(vibeBckFrame, vibeBckFrame.rows/10, vibeBckFrame.cols/20);
		displayFrame("sWNDVibe1", sWNDVibe); 

		//performing sWND
		sWNDVibe = slidingWindowNeighborDetector(vibeBckFrame, vibeBckFrame.rows/20, vibeBckFrame.cols/40);
		displayFrame("sWNDVibe2", sWNDVibe); 

		//performing canny
		Mat sWNDVibeCanny = cannyContourDetector(sWNDVibe);
		displayFrame("sWNDVibeCannycanny2", sWNDVibeCanny);
	}

	//returning processed frame
	return sWNDVibeCanny;
}


//method to do background subtraction with MOG 1
Mat bgMog(bool buffer)
{

	//instantiating Mat objects
	Mat fgmask;
	Mat bck;
	Mat fgMaskSWNDCanny;
	Mat fgmaskSWND;

	//performing background subtraction
    bckSubMOG.operator()(globalFrames.at(i), fgmask, .01); //1.0 / 200);

	if(!buffer)
	{ 
		displayFrame("MOG Fg MAsk", fgmask);
 		//displayFrame("RCFrame", globalFrames[i]);

		cout << " df" << endl;
 		//performing sWND
		fgmaskSWND = slidingWindowNeighborDetector(fgmask, fgmask.rows/10, fgmask.cols/20);
		//displayFrame("fgmaskSWND", fgmaskSWND);
 			cout << "here" << endl;

		fgmaskSWND = slidingWindowNeighborDetector(fgmaskSWND, fgmaskSWND.rows/20, fgmaskSWND.cols/40);
		//displayFrame("fgmaskSWNDSWND2", fgmaskSWND);

		fgmaskSWND = slidingWindowNeighborDetector(fgmaskSWND, fgmaskSWND.rows/30, fgmaskSWND.cols/60);
		//displayFrame("fgmaskSWNDSWND3", fgmaskSWND);

		//performing canny
		fgMaskSWNDCanny = cannyContourDetector(fgmaskSWND);
		//displayFrame("fgMaskSWNDCanny2", fgMaskSWNDCanny);
	}



	//return canny
	return fgMaskSWNDCanny;
}


//method to handle OFA thread
Mat opticalFlowFarneback()
{
	//cout << "ENTERING OFF" << endl;
	//instantiate thread object
	pthread_t opticalFlowFarneback;

	//instantiating multithread Data object
	struct thread_data threadData;

	//saving data to pass
	threadData.data = i;

	//create OFA thread
	pthread_create(&opticalFlowFarneback, NULL, computeOpticalFlowAnalysisThread, (void *)&threadData);
	
	while(opticalFlowThreadCompletion != 1){}

	opticalFlowThreadCompletion = 0;

	return thresholdFrameOFA;
}

//write initial statistics about the video
void writeInitialStats(int NUMBER_OF_FRAMES, int FRAME_RATE, int FRAME_WIDTH, int FRAME_HEIGHT, const char* filename)
{
	////writing stats to txt file
	//initiating write stream
	ofstream writeToFile;

	//creating filename  ending
	string filenameAppend = "Stats.txt";

	//concanating and creating file name string
	string strFilename = filename + currentDateTime() + filenameAppend;

	//open file stream and begin writing file
	writeToFile.open (strFilename);

	//write video statistics
	writeToFile << "Stats on video >> There are = " << NUMBER_OF_FRAMES << " frames. The frame rate is " << FRAME_RATE
	<< " frames per second. Resolution is " << FRAME_WIDTH << " X " << FRAME_HEIGHT;

	//close file stream
	writeToFile.close();

	if(debug)
	{
		//display video statistics
		cout << "Stats on video >> There are = " << NUMBER_OF_FRAMES << " frames. The frame rate is " << FRAME_RATE
				<< " frames per second. Resolution is " << FRAME_WIDTH << " X " << FRAME_HEIGHT << endl;;
	}
}

//display welcome message and splash screen
void welcome()
{
	if(i < bufferMemory * 2)
	{
 		Mat img = imread("assets/TCD3.png");
		putText(img, "Initializing; V. Prasad 2015 All Rights Reserved"
			, cvPoint(30,30),CV_FONT_HERSHEY_SIMPLEX, .75, cvScalar(255,255,0), 1, CV_AA, false);

		//display welcome images
		displayFrame("Welcome", img, true);
	}

	else
	{
		//close welcome image
		destroyWindow("Welcome");
	}
}

//calculate time for each iteration
double calculateFPS(clock_t tStart, clock_t tFinal)
{
	//return frames per second
	return 1/((((float)tFinal-(float)tStart) / CLOCKS_PER_SEC));
}

//method to calculate runtime
void computeRunTime(clock_t t1, clock_t t2, int framesRead)
{
	//subtract from start time
	float diff ((float)t2-(float)t1);

	//calculate frames per second
	double frameRateProcessing = (framesRead / diff) * CLOCKS_PER_SEC;

	//display amount of time for run time
	cout << (diff / CLOCKS_PER_SEC) << " seconds of run time." << endl;

	//display number of frames processed per second
	cout << frameRateProcessing << " frames processed per second." << endl;
	cout << framesRead << " frames read." << endl;
}

//method to calculate median of vector of integers
double calcMedian(vector<int> integers)
{
	//double to store non-int median
	double median;

	//read size of vector
	size_t size = integers.size();

	//sort array
    sort(integers.begin(), integers.end());

    //if even number of elements
	if (size % 2 == 0)
	{
		//median is middle elements averaged
		median = (integers[size / 2 - 1] + integers[size / 2]) / 2;
	}

	//if odd number of elements
	else
	{
		//median is middle element
		median = integers[size / 2];
	}

	//return the median value
	return median;
}

//method to calculate mean of vector of integers
double calcMean(vector <int> integers)
{
	//total of all elements
	int total = 0;

	//step through vector
	for (int v = 0; v < integers.size(); v++)
	{
		//total all values
		total += integers.at(v);
	}

	//return mean value
	return total/integers.size();
}

//method to identify type of Mat based on identifier
string type2str(int type) {

	//string to return type of mat
	string r;

	//stats about frame
	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	//switch to determine Mat type
	switch ( depth ) {
		case CV_8U:  r = "8U"; break;
		case CV_8S:  r = "8S"; break;
		case CV_16U: r = "16U"; break;
		case CV_16S: r = "16S"; break;
		case CV_32S: r = "32S"; break;
		case CV_32F: r = "32F"; break;
		case CV_64F: r = "64F"; break;
		default:     r = "User"; break;
    }

	//append formatting
	r += "C";
	r += (chans+'0');

	//return Mat type
	return r;
}

//thread to calculate median of image
void *calcMedianImage(void *threadarg)
{
	//defining data structure to read in info to new thread
	struct thread_data *data;
	data = (struct thread_data *) threadarg;

	//performing deep copy
	globalGrayFrames[i].copyTo(backgroundFrameMedian);

	//variables to display completion
	double displayPercentageCounter = 0;
	double activeCounter = 0;

	//calculating number of runs
	for(int j=0;j<backgroundFrameMedian.rows;j++)
	{
		for (int a=0;a<backgroundFrameMedian.cols;a++)
		{
			for (int t = (i - bufferMemory); t < i ; t++)
			{
				displayPercentageCounter++;
			}
		}
	}

	//stepping through all pixels
	for(int j=0;j<backgroundFrameMedian.rows;j++)
	{
		for (int a=0;a<backgroundFrameMedian.cols;a++)
		{
			//saving all pixel values
			vector <int> pixelHistory;

			//moving through all frames stored in buffer
			for (int t = (i - bufferMemory); t < i ; t++)
			{
				//Mat to store current frame to process
				Mat currentFrameForMedianBackground;

				//copy current frame
				globalGrayFrames.at(i-t).copyTo(currentFrameForMedianBackground);

				//save pixel into pixel history
				pixelHistory.push_back(currentFrameForMedianBackground.at<uchar>(j,a));

				//increment for load calculations
				activeCounter++;
			}

			//calculate median value and store in background image
			backgroundFrameMedian.at<uchar>(j,a) = calcMedian(pixelHistory);
	   }

	   //display percentage completed
 	   cout << ((activeCounter / displayPercentageCounter) * 100) << "% Median Image Scanned" << endl;

	}

  	backgroundFrameMedian.copyTo(finalTrackingFrame);

	//signal thread completion
    medianImageCompletion = 1;
}

//calculate max value in frame for debug
int maxMat(Mat sourceFrame)
{
	//variable for current max
	int currMax = INT_MIN;

	//step through pixels
	for(int j=0;j<sourceFrame.rows;j++)
	{
	    for (int a=0;a<sourceFrame.cols;a++)
	    {
	    	//if current value is larger than previous max
	    	if(sourceFrame.at<uchar>(j,a) > currMax)
	    	{
	    		//store current value as new max
	    		currMax = sourceFrame.at<uchar>(j,a);
	    	}
	    }
	}

	//return max value in matrix
	return currMax;
}

//method to threshold standard frame
Mat thresholdFrame(Mat sourceDiffFrame, const int threshold)
{
	//Mat to hold frame
	Mat thresholdFrame;

	//perform deep copy into destination Mat
	sourceDiffFrame.copyTo(thresholdFrame);

	//steping through pixels
	for(int j=0;j<sourceDiffFrame.rows;j++)
	{
	    for (int a=0;a<sourceDiffFrame.cols;a++)
	    {
	    	//if pixel value greater than threshold
	    	if(sourceDiffFrame.at<uchar>(j,a) > threshold)
	    	{
	    		//write to binary image
	    		thresholdFrame.at<uchar>(j,a) = 255;
	    	}
	    	else
	    	{
	    		//write to binary image
	    		thresholdFrame.at<uchar>(j,a) = 0;
	    	}
	    }
	}

	//perform morphology
	//thresholdFrame = morph(thresholdFrame, 1, "closing");

	//return thresholded frame
	return thresholdFrame;
}

//method to perform simple image subtraction
Mat imageSubtraction()
{
  	//subtract frames
	Mat tmpStore  =  globalGrayFrames[i] - backgroundFrameMedian;

	displayFrame("Raw imgSub", tmpStore);
 	//threshold frames
	tmpStore = thresholdFrame(tmpStore, 50);
	displayFrame("Thresh imgSub", tmpStore);

	//perform sWND
	tmpStore =  slidingWindowNeighborDetector(tmpStore, tmpStore.rows / 5, tmpStore.cols / 10);
 	displayFrame("SWD",tmpStore);
 	tmpStore = slidingWindowNeighborDetector(tmpStore, tmpStore.rows / 10, tmpStore.cols / 20);
 	displayFrame("SWD2", tmpStore);
 	tmpStore = slidingWindowNeighborDetector(tmpStore, tmpStore.rows / 20, tmpStore.cols / 40);
	displayFrame("SWD3", tmpStore);

	//perform canny
	tmpStore = cannyContourDetector(tmpStore);
	displayFrame("Canny Contour", tmpStore);

	//return frame
	return tmpStore;
}

//method to perform median on grayscale images
void grayScaleFrameMedian()
{
	if(debug)
		cout << "Entered gray scale median" << endl;

	//instantiating multithread object
	pthread_t medianImageThread;

	//instantiating multithread Data object
	struct thread_data threadData;

	//saving data into multithread
	threadData.data = i;

	//creating thread to calculate median of image
	pthread_create(&medianImageThread, NULL, calcMedianImage, (void *)&threadData);
  
	//save median image
	imwrite((currentDateTime() + "medianBackgroundImage.jpg"), backgroundFrameMedian);
}

//method to calculate Gaussian image difference
void *calcGaussianMixtureModel(void *threadarg)
{
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

	displayFrame("GMM Binary Frame" , binaryGMMFrame);

	//if buffer built
	if(i > bufferMemory * 2)
	{
		//perform sWND
		gmmFrame = slidingWindowNeighborDetector(binaryGMMFrame, gmmFrame.rows / 5, gmmFrame.cols / 10);
		displayFrame("sWDNs GMM Frame 1", gmmFrame);

		gmmFrame = slidingWindowNeighborDetector(gmmFrame, gmmFrame.rows / 10, gmmFrame.cols / 20);
		displayFrame("sWDNs GMM Frame 2", gmmFrame);

		gmmFrame = slidingWindowNeighborDetector(gmmFrame, gmmFrame.rows / 20, gmmFrame.cols / 40);
		displayFrame("sWDNs GMM Frame 3", gmmFrame);

		Mat gmmFrameSWNDCanny = gmmFrame;

		if(i > bufferMemory * 3 -1)
		{
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
Mat gaussianMixtureModel()
{
 
	//instantiate thread object
	pthread_t gaussianMixtureModelThread;

	//instantiating multithread Data object
	struct thread_data threadData;

	//save i data
	threadData.data = i;

	//create thread
	pthread_create(&gaussianMixtureModelThread, NULL, calcGaussianMixtureModel, (void *)&threadData);
  	
  	//return processed frame if completed
	if(i > bufferMemory * 2)
		return cannyGMM;
	//return tmp frame if not finished
	else
		return globalFrames[i];
}


//method to handle all background image generation
void generateBackgroundImage(int FRAME_RATE)
{
	//if post-processing
	if(readMedianImg && useMedians && i < bufferMemory + 5)
	{
		//read median image
		backgroundFrameMedian = imread("assets/froggerHighwayDrunkMedian.jpg");

		//convert to grayscale
		cvtColor(backgroundFrameMedian, backgroundFrameMedian, CV_BGR2GRAY);

		displayFrame("backgroundFrameMedian", backgroundFrameMedian);

	 	backgroundFrameMedian.copyTo(finalTrackingFrame);
	}

	//if real-time calculation
	else
	{
		//after initial buffer read and using medians
		if(i == bufferMemory && useMedians)
		{ 
 			grayScaleFrameMedian();

 			while(medianImageCompletion != 1) {}
		}
		//every 3 minutes
		if (i % (FRAME_RATE * 180) == 0 && i > 0)
		{
			//calculate new medians
 			grayScaleFrameMedian();

 			while(medianImageCompletion != 1) {}

		}
	}


	medianImageCompletion = 0;
}

//method to draw canny contours
Mat cannyContourDetector(Mat srcFrame)
{
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
	Canny(srcFrame , cannyFrame, 300, 900, 3);
 	findContours(cannyFrame, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
	
	//creating blank frame to draw on
	Mat drawing = Mat::zeros( cannyFrame.size(), CV_8UC3 );
	
	//moments for center of mass
    vector<Moments> mu(contours.size() );
    for( int i = 0; i < contours.size(); i++ )
       { mu[i] = moments( contours[i], false ); }

	//get mass centers:
    vector<Point2f> mc( contours.size() );
    for( int i = 0; i < contours.size(); i++ )
       { mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 ); }
	 
   	//for each detected contour
 	for(int v = 0; v < contours.size(); v++)
	{
		//if large enough to be object
  		if(arcLength(contours[v], true) > thresholdNoiseSize && arcLength(contours[v], true)  < misDetectLargeSize)
 		{
 			//draw object and circle center point
			drawContours( drawing, contours, v, Scalar(254,254,0), 2, 8, hierarchy, 0, Point() );
			circle( drawing, mc[v], 4, Scalar(254, 254, 0), -1, 8, 0 );
			fillCoordinates(mc);
 		}
 	}

 	//return image with contours
	return drawing;
}

Mat slidingWindowNeighborPointDetector (Mat sourceFrame, int numRowSections, int numColumnSections, vector <Point> coordinates)
{
	//if using default num rows
		if(numRowSections == -1 || numColumnSections == -1)
		{
			//split into standard size
			numRowSections = sourceFrame.rows / 10;
			numColumnSections = sourceFrame.cols / 20;
		}

		/*

		double numRowSectionsDouble = numRowSections;
		double numColumnSectionsDouble = numColumnSections;

		while(sourceFrame.rows % numRowSections != 0)
		{
			numRowSections++;
			numRowSectionsDouble = numRowSections;
		}

		while(sourceFrame.cols % numColumnSections != 0)
		{
			numColumnSections++;
			numColumnSectionsDouble = numColumnSections;
		}

		*/

		//declaring percentage to calculate density
		double percentage = 0;

		//setting size of search area
		int windowWidth = sourceFrame.rows / numRowSections;
		int windowHeight = sourceFrame.cols / numColumnSections;

		//creating destination frame of correct size
		Mat destinationFrame = Mat(sourceFrame.rows, sourceFrame.cols, CV_8UC1);

		//cycling through pieces
		for(int v = windowWidth/2; v <= sourceFrame.rows - windowWidth/2; v++)
		{
			for(int j = windowHeight/2; j <= sourceFrame.cols - windowHeight/2; j++)
			{
				/*
				//variables to calculate density
				double totalCounter = 0;
				double detectCounter = 0;
				*/

				int pointsCounter = 0;

				//cycling through neighbors
				for(int x =  v - windowWidth/2; x < v + windowWidth/2; x++)
				{
					for(int k = j - windowHeight/2; k < j + windowHeight/2; k++)
					{
						/*
						int z = 0;
						int pointsCounter = 0;
						while(pointsCounter <= 1 && z < coordinates.size())
						{
							for(int b = 0; b < coordinates.size())
						}
						*/

						Point currentPoint(x,k);

						for(int z = 0; z< coordinates.size(); z++)
						{
							if(coordinates[z] == currentPoint)
							{
								pointsCounter++;
							}
						}

						/*
						//if object exists
						if(sourceFrame.at<uchar>(x,k) > 127)
						{
							//add to detect counter
							detectCounter++;
						}

						//count pixels searched
						totalCounter++;
						*/
					}
				}

				/*
				//prevent divide by 0 if glitch and calculate percentage
				if(totalCounter != 0)
					percentage = detectCounter / totalCounter;
				else
					cout << "Ted Cruz" << endl;
				*/

				//if object exists flag it
				if(pointsCounter >  1)
				{
	 				destinationFrame.at<uchar>(v,j) = 255;
				}

				//else set it to 0
				else
				{
					//sourceFrame.at<uchar>(v,j) = 0;
					destinationFrame.at<uchar>(v,j) = 0;
				}
			}
		}

		//return processed frame
	 	return destinationFrame;
}

//method to perform proximity density search to remove noise and identify noise
Mat slidingWindowNeighborDetector(Mat sourceFrame, int numRowSections, int numColumnSections)
{
	//if using default num rows
	if(numRowSections == -1 || numColumnSections == -1)
	{
		//split into standard size
		numRowSections = sourceFrame.rows / 10;
		numColumnSections = sourceFrame.cols / 20;
	} 

	/*

	double numRowSectionsDouble = numRowSections;
	double numColumnSectionsDouble = numColumnSections;

	while(sourceFrame.rows % numRowSections != 0)
	{
		numRowSections++;
		numRowSectionsDouble = numRowSections;
	}

	while(sourceFrame.cols % numColumnSections != 0)
	{
		numColumnSections++;
		numColumnSectionsDouble = numColumnSections;
	}

	*/

	//declaring percentage to calculate density
	double percentage = 0;

	//setting size of search area
	int windowWidth = sourceFrame.rows / numRowSections;
	int windowHeight = sourceFrame.cols / numColumnSections;

	//creating destination frame of correct size
	Mat destinationFrame = Mat(sourceFrame.rows, sourceFrame.cols, CV_8UC1);

	//cycling through pieces
	for(int v = windowWidth/2; v <= sourceFrame.rows - windowWidth/2; v++)
	{
		for(int j = windowHeight/2; j <= sourceFrame.cols - windowHeight/2; j++)
		{
			//variables to calculate density
			double totalCounter = 0;
			double detectCounter = 0;
 
 			//cycling through neighbors
			for(int x =  v - windowWidth/2; x < v + windowWidth/2; x++)
			{
				for(int k = j - windowHeight/2; k < j + windowHeight/2; k++)
				{
					//if object exists
					if(sourceFrame.at<uchar>(x,k) > 127)
					{
						//add to detect counter
						detectCounter++;
					}

					//count pixels searched
					totalCounter++;
				}
			}

			//prevent divide by 0 if glitch and calculate percentage
			if(totalCounter != 0)
				percentage = detectCounter / totalCounter;
			else
				cout << "Ted Cruz" << endl;

			//if object exists flag it
			if(percentage >  .25)
			{
 				destinationFrame.at<uchar>(v,j) = 255;
			}

			//else set it to 0
			else
			{
				//sourceFrame.at<uchar>(v,j) = 0;
				destinationFrame.at<uchar>(v,j) = 0;
			}
		}
	}

	//return processed frame
 	return destinationFrame;
}


//method to handle median image subtraction
Mat medianImageSubtraction(int FRAME_RATE)
{
	//generate or read background image
	generateBackgroundImage(FRAME_RATE);

	//calculate image difference and return
	return imageSubtraction();
}

//method to perform vibe background subtraction
void *computeVibeBackgroundThread(void *threadarg)
{
	struct thread_data *data;
	data = (struct thread_data *) threadarg;

	//instantiating Mat frame object
	Mat sWNDVibeCanny;

	//if done buffering
	if(i == bufferMemory)
	{
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

	else
	{
		//instantiating Mat frame object
		Mat resizedFrame;

		//saving current frame
		globalFrames[i].copyTo(resizedFrame); 

		//processing model
		vibeBckFrame = *bgfg.fg(resizedFrame);

		displayFrame("vibeBckFrame", vibeBckFrame);

		//performing sWND
		Mat sWNDVibe = slidingWindowNeighborDetector(vibeBckFrame, vibeBckFrame.rows/10, vibeBckFrame.cols/20);
		displayFrame("sWNDVibe1", sWNDVibe);

		//performing sWND
		sWNDVibe = slidingWindowNeighborDetector(vibeBckFrame, vibeBckFrame.rows/20, vibeBckFrame.cols/40);
		displayFrame("sWNDVibe2", sWNDVibe);

		Mat sWNDVibeCanny = sWNDVibe;

		if(i > bufferMemory * 3 -1)
		{
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

void vibeBackgroundSubtractionThreadHandler(bool buffer)
{
	//instantiating multithread object
	pthread_t vibeBackgroundSubtractionThread;
	
	//instantiating multithread Data object
	struct thread_data threadData;		

	//saving data into data object
	threadData.data = i;

	//creating threads
	int vibeBackgroundThreadRC = pthread_create(&vibeBackgroundSubtractionThread, NULL, computeVibeBackgroundThread, (void *)&threadData);	
}

//method to do background subtraction with MOG 1
void *computeBgMog1(void *threadarg)
{
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
	Mat fgmaskSWND = slidingWindowNeighborDetector(fgmask, fgmask.rows/10, fgmask.cols/20);
	displayFrame("fgmaskSWND", fgmaskSWND);

	fgmaskSWND = slidingWindowNeighborDetector(fgmaskSWND, fgmaskSWND.rows/20, fgmaskSWND.cols/40);
	displayFrame("fgmaskSWNDSWND2", fgmaskSWND);

	fgmaskSWND = slidingWindowNeighborDetector(fgmaskSWND, fgmaskSWND.rows/30, fgmaskSWND.cols/60);
	displayFrame("fgmaskSWNDSWND3", fgmaskSWND);

	//performing canny
	fgMaskSWNDCanny = cannyContourDetector(fgmaskSWND);
	displayFrame("fgMaskSWNDCanny2", fgMaskSWNDCanny);

	//return canny
	mogDetection1GlobalFrame = fgMaskSWNDCanny;

	//signal completion
	mogDetection1GlobalFrameCompletion = 1;
}


void mogDetectionThreadHandler(bool buffer)
{
	//instantiating multithread object
	pthread_t mogDetectionThread;
	
	//instantiating multithread Data object
	struct thread_data threadData;		

	//saving data into data object
	threadData.data = i;

	//creating threads
	int mogDetectionThreadRC = pthread_create(&mogDetectionThread, NULL, computeBgMog1, (void *)&threadData);	
}

//method to do background subtraction with MOG 1
void *computeBgMog2(void *threadarg)
{
	struct thread_data *data;
	data = (struct thread_data *) threadarg;

	//instantiating Mat objects
	Mat fgmaskShadow;
	Mat frameToResizeShadow;
	
 	//copying into tmp variable
 	globalFrames[i].copyTo(frameToResizeShadow);
 
 	//performing background subtraction 
	pMOG2Shadow->operator()(frameToResizeShadow , fgmaskShadow, .01);

	//performing sWND
	displayFrame("fgmaskShadow" , fgmaskShadow);
	Mat fgmaskShadowSWND = slidingWindowNeighborDetector(fgmaskShadow, fgmaskShadow.rows/10, fgmaskShadow.cols/20);
	displayFrame("fgmaskShadowSWND", fgmaskShadowSWND); 

	fgmaskShadowSWND = slidingWindowNeighborDetector(fgmaskShadowSWND, fgmaskShadowSWND.rows/20, fgmaskShadowSWND.cols/40);
	displayFrame("fgmaskShadowSWND2", fgmaskShadowSWND);

	//performing canny
	Mat fgMaskShadowSWNDCanny = cannyContourDetector(fgmaskShadowSWND);
	displayFrame("fgMaskShadowSWNDCanny2", fgMaskShadowSWNDCanny);
 
	//return canny
	mogDetection2GlobalFrame = fgMaskShadowSWNDCanny;

	//signal completion
	mogDetection2GlobalFrameCompletion = 1;
}

void mogDetection2ThreadHandler(bool buffer)
{
	//instantiating multithread object
	pthread_t mogDetection2Thread;
	
	//instantiating multithread Data object
	struct thread_data threadData;		

	//saving data into data object
	threadData.data = i;

	//creating threads
	int mogDetection2ThreadRC = pthread_create(&mogDetection2Thread, NULL, computeBgMog2, (void *)&threadData);
}


//method to handle median image subtraction
void *computeMedianDetection(void *threadarg)
{
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

void medianDetectionThreadHandler(int FRAME_RATE)
{
	//instantiating multithread object
	pthread_t medianDetectionThread;
	
	//instantiating multithread Data object
	struct thread_data threadData;		

	//saving data into data object
	threadData.data = FRAME_RATE;

	//creating threads
	int medianDetectionThreadRC = pthread_create(&medianDetectionThread, NULL, computeMedianDetection, (void *)&threadData);
}


//method to handle all image processing object detection
void objectDetection(int FRAME_RATE)
{
 	//save all methods vehicle canny outputs
 	Mat gmmDetection = gaussianMixtureModel();

	//Mat tmpMedian = medianImageSubtraction(FRAME_RATE);

	Mat ofaDetection = opticalFlowFarneback();
 
	//Mat ofaDetection;
	//vibeDetectionGlobalFrame = vibeBackgroundSubtraction(false);
 	//Mat tmpMOG1 =  bgMog(false);
	//Mat tmpMOG2 = bgMog2(false); 

	//opticalFlowFarneback();
 	vibeBackgroundSubtractionThreadHandler(false);
 	mogDetectionThreadHandler(false);
 	mogDetection2ThreadHandler(false);
 	medianDetectionThreadHandler(FRAME_RATE);

  	bool firstTimeMedianImage = true;
 	bool firstTimeVibe = true;
 	bool firstTimeMOG1 = true;
 	bool firstTimeMOG2 = true;
 	bool enterOnce = true;

 	Mat tmpMedian;
 	Mat tmpVibe;
 	Mat tmpMOG1;
 	Mat tmpMOG2;

 	/*
	while(medianDetectionGlobalFrameCompletion != 1||
 		vibeDetectionGlobalFrameCompletion != 1||
		mogDetection1GlobalFrameCompletion != 1||
		mogDetection2GlobalFrameCompletion != 1 ||
		enterOnce
		)
	*/

 	bool finishedMedian = false;
 	bool finishedVibe = false;
 	bool finishedMOG1 = false;
 	bool finishedMOG2 = false;

	while(!finishedMedian ||
			!finishedVibe ||
			!finishedMOG1||
			!finishedMOG2 ||
			enterOnce
			)
	{
 		enterOnce = false;

		if(firstTimeMedianImage && medianDetectionGlobalFrameCompletion == 1 )
		{
			tmpMedian = medianDetectionGlobalFrame;
			displayFrame("medianDetection", tmpMedian);
			firstTimeMedianImage = false;
			finishedMedian = true;
		}
		if(firstTimeVibe && vibeDetectionGlobalFrameCompletion == 1 )
		{
			tmpVibe = vibeDetectionGlobalFrame;
			displayFrame("vibeDetection", tmpVibe);
			firstTimeVibe = false;
			finishedVibe = true;
		}
		if(firstTimeMOG1 && mogDetection1GlobalFrameCompletion == 1 )
		{
			tmpMOG1= mogDetection1GlobalFrame;
			displayFrame("mogDetection1", tmpMOG1);
			firstTimeMOG1 = false;
			finishedMOG1 = true;
		}
		if(firstTimeMOG2 && mogDetection2GlobalFrameCompletion == 1)
		{
			tmpMOG2 = mogDetection2GlobalFrame;
			displayFrame("mogDetection2", tmpMOG2);
			firstTimeMOG2 = false;
			finishedMOG2 = true;
		}
 	}
 	
	vibeDetectionGlobalFrameCompletion = 0;
	mogDetection1GlobalFrameCompletion = 0;
	mogDetection2GlobalFrameCompletion = 0;
 	medianDetectionGlobalFrameCompletion = 0;


	displayFrame("vibeDetection", tmpVibe, true);
	displayFrame("mogDetection1", tmpMOG1, true);
	displayFrame("mogDetection2", tmpMOG2, true);
	displayFrame("medianDetection", tmpMedian, true);
	displayFrame("gmmDetection", gmmDetection);
	displayFrame("ofaDetection", ofaDetection, true);
	displayFrame("Raw Frame", globalFrames[i]);

 	waitKey(30);

  	if(i > bufferMemory + 5 && tmpMOG1.channels() == 3 && tmpMOG2.channels() == 3 && ofaDetection.channels() == 3 && tmpMedian.channels() == 3)
 	{
   		if(i > bufferMemory * 3 + 2 && tmpMOG1.channels() == 3 && tmpMOG2.channels() == 3 && ofaDetection.channels() == 3 && tmpMedian.channels() == 3
  				&& tmpVibe.channels() == 3 && gmmDetection.channels() == 3 && 1 == 2)
  		{
  			Mat combined = tmpMOG1 +  tmpMOG2 + ofaDetection + tmpMedian + tmpVibe + gmmDetection;
			displayFrame("Combined Contours", combined);

			double beta = ( 1.0 - .5 );
			addWeighted( combined, .5, globalFrames[i], beta, 0.0, combined);
			displayFrame("Overlay", combined, true);
  		}
  		else
  		{
			Mat combined = tmpMOG1 +  tmpMOG2 + ofaDetection + tmpMedian;
			displayFrame("Combined Contours", combined);

			double beta = ( 1.0 - .5 );
			addWeighted( combined, .5, globalFrames[i], beta, 0.0, combined);
			displayFrame("Overlay", combined, true);
  		}
 	}

 	else
 	{
 		cout << "Sync Issue" << endl;
 	}

}

void fillCoordinates(vector <Point2f> detectedCoordinatesMoments)
{
 	for(int v = 0; v < detectedCoordinatesMoments.size(); v++)
	{
		Point tmpPoint ((int) detectedCoordinatesMoments[v].x,  (int) detectedCoordinatesMoments[v].y);

		if((tmpPoint.x > 30 && tmpPoint.x < globalGrayFrames[i].cols - 60) &&
				(tmpPoint.y > 30 && tmpPoint.y < globalGrayFrames[i].rows - 30))
 		{
			detectedCoordinates.push_back(tmpPoint);
 		}
 	}
}

bool point_comparator(const cv::Point2f &a, const cv::Point2f &b) {
    return a.x*a.x + a.y*a.y < b.x*b.x + b.y*b.y; // (/* Your expression */);
}

void displayCoordinates(vector <Point> coordinatesToDisplay)
{
 	for(int v = 0; v < coordinatesToDisplay.size(); v++)
	{
		cout << "(" << coordinatesToDisplay[v].x << "," << coordinatesToDisplay[v].y << ")" << endl;
	}

}

Mat checkBlobVotes(Mat srcFrame, vector <Point> coordinates)
{
	return srcFrame;
}

void drawCoordinates(vector <Point> coordinatesToDisplay, String initialName)
{
	//Mat tmpToDraw;

	//globalFrames[i].copyTo(tmpToDraw);

	Mat tmpToDraw = Mat::zeros(globalFrames[i].size(), CV_16UC3);

	/*
	for(int v = 0; v < tmpToDraw.rows * 1; v++)
	{
		for(int j = 0; j < tmpToDraw.cols; j++)
		{
			tmpToDraw.at<uchar>(v,j) = 0;
		}
	}
	*/

	cout << " SIZE OF COORDINATES TO DISPLAY " << coordinatesToDisplay.size() << endl;

	for(int v = 0; v < coordinatesToDisplay.size(); v++)
	{
		circle( tmpToDraw, coordinatesToDisplay[v], 4, Scalar(254, 254, 0), -1, 8, 0 );
	}

	//destroyWindow("Coordinates to Display");
	displayFrame(initialName ,tmpToDraw, true);
	//imwrite("trackingFrame" + initialName + ".TIFF", trackingFrame);

	//tmpToDraw = slidingWindowNeighborPointDetector(tmpToDraw, tmpToDraw.rows / 5, tmpToDraw.cols / 10 , coordinatesToDisplay);
	//**tmpToDraw = slidingWindowNeighborPointDetector(tmpToDraw, tmpToDraw.rows / 10, tmpToDraw.cols / 20 , coordinatesToDisplay);
	//tmpToDraw = slidingWindowNeighborPointDetector(tmpToDraw, tmpToDraw.rows / 20, tmpToDraw.cols / 40 , coordinatesToDisplay);
	//tmpToDraw = slidingWindowNeighborPointDetector(tmpToDraw, tmpToDraw.rows / 30, tmpToDraw.cols / 60 , coordinatesToDisplay);
	//tmpToDraw = slidingWindowNeighborPointDetector(tmpToDraw, tmpToDraw.rows / 40, tmpToDraw.cols / 80 , coordinatesToDisplay);

	//tmpToDraw = checkBlobVotes(tmpToDraw, coordinatesToDisplay);

	//displayFrame("sWNPD Frame", tmpToDraw, true);

}

vector <Point> sortCoordinates(vector <Point> coordinates)
{
	sort(coordinates.begin(), coordinates.end(), point_comparator);
	return coordinates;
}

Point averagePoints(vector <Point> coordinates)
{
	if(coordinates.size() == 0)
	{
		double xCoordinate = 0;
		double yCoordinate = 0;

		for(int v = 0; v < coordinates.size(); v++)
		{
			xCoordinate += coordinates[v].x;
			yCoordinate += coordinates[v].y;
		}

		Point tmpPoint(xCoordinate/coordinates.size(), yCoordinate/coordinates.size());

		return tmpPoint;
	}

	else
	{
		return coordinates[0];
	}
}

vector <Point> averageCoordinates(vector <Point> coordinates, int distanceThreshold)
{
	if(coordinates.size() > 1)
	{
		vector <Point> destinationCoordinates;
		vector <Point> pointsToAverage;
		coordinates = sortCoordinates(coordinates);
		Point tmpPoint = coordinates[0];

		bool enteredOnce = false;

		for(int v = 0; v < coordinates.size(); v++)
		{
			double tmp1 = abs( tmpPoint.y - coordinates[v].y);
			double tmp2 =  abs(tmpPoint.x - coordinates[v].x);
			double tmp = sqrt(tmp1 * tmp2);
			/*
			cout << tmp1 << " tmp1 " << endl;
			cout << tmp2 << " tmp2 " << endl;
			cout << tmp << " tmp " << endl;
			*/
			if(sqrt((abs(tmpPoint.y - coordinates[v].y) * (abs(tmpPoint.x - coordinates[v].x)))) > distanceThreshold)
			{
				//cout << "Entered Refresh " << v << endl;
				destinationCoordinates.push_back(averagePoints(pointsToAverage));
				tmpPoint = coordinates[v];
				pointsToAverage.erase(pointsToAverage.begin(), pointsToAverage.end());
				bool enteredOnce =  true;
			}
			else
			{
				//cout << "Entered Old " << v << endl;
				pointsToAverage.push_back(coordinates[v]);
			}
		}

		if(!enteredOnce)
		{
			destinationCoordinates.push_back(averagePoints(pointsToAverage));
		}

		else if(pointsToAverage.size() > 0)
		{
			destinationCoordinates.push_back(averagePoints(pointsToAverage));
		}

		return destinationCoordinates;
	}

	else
	{
 		return coordinates;
	}
}

void drawAllTracking()
{
 	for(int v = 0; v < detectedCoordinates.size(); v++)
	{
 		circle(	finalTrackingFrame, detectedCoordinates[v], 4, Scalar(254, 254, 0), -1, 8, 0 );
 	}
 	displayFrame("All Tracking Frame", finalTrackingFrame, true);
 }

void registerFirstCar()
{
	vectorOfDetectedCars.push_back(detectedCoordinates);

	for(int v= 0 ; v< detectedCoordinates.size(); v++)
	{
		if(detectedCoordinates[v].x < 75)
		{
 			vector <Point> carCoordinate;
			carCoordinate.push_back(detectedCoordinates[v]);
			carCoordinates.push_back(carCoordinate);
		}
 	}
}

bool checkBasicXYAnomaly(int xMovement, int yMovement, Point carPoint)
{
	const double maxThreshold = 10;
	const double minThreshold = -10;

	cout << "X MOVEMENT " << (double) xMovement << endl;
	cout << "Y MOVEMENT " << (double) yMovement << endl;

	cout << "X LEARNED " << (double) xLearnedMovement << endl;
	cout << "Y LEARNED " << (double) yLearnedMovement << endl;

  	Mat drawAnomalyCar = Mat::zeros( globalFrames[i].size(), CV_16UC3 );

	if(((xMovement > xLearnedMovement + maxThreshold || xMovement < xLearnedMovement + minThreshold))
			|| ((yMovement > yLearnedMovement + maxThreshold) || (yMovement < yLearnedMovement + minThreshold)))
	{
		circle(drawAnomalyCar, carPoint, 5, Scalar(0, 0, 255), -1);
		cout << " !!!!!!!!!!!ANOMALY DETECTED!!!!!!!!!!!" << endl;
	}
	displayFrame("drawAnomalyCar", drawAnomalyCar, true);
}

int findMin(int num1, int num2)
{
	if(num1 < num2)
	{
		return num1;
	}
	else if(num2 < num1)
	{
		return num2;
	}
	else
	{
		return num1;
	}
}

void analyzeMovement()
{
	vector <Point> currentDetects =  vectorOfDetectedCars[vectorOfDetectedCars.size() - 1];
	vector <Point> prevDetects =  vectorOfDetectedCars[vectorOfDetectedCars.size() - 2];

	const int distanceThreshold = 15;

	int least = findMin(currentDetects.size(), prevDetects.size());

	for(int v = 0; v < least; v++)
	{
		currentDetects = sortCoordinates(currentDetects);
		prevDetects = sortCoordinates(prevDetects);

		double lowestDistance = INT_MAX;
		double distance;

		Point tmpPoint;
		Point tmpDetectPoint;
		Point bestPoint;

		for(int j = 0; j < prevDetects.size(); j++ )
		{
			tmpDetectPoint = prevDetects[j];
			tmpPoint = currentDetects[v];

			distance = sqrt(abs(tmpDetectPoint.x - tmpPoint.x) * (abs(tmpDetectPoint.y - tmpPoint.y)));

			if(distance < lowestDistance)
			{
				lowestDistance = distance;
				bestPoint  = tmpDetectPoint;
 			}
			//carCoordinates.push_back();
		}

		int xDisplacement = abs(bestPoint.x - tmpPoint.x);
		int yDisplacement = abs(bestPoint.y - tmpPoint.y);

		if(lowestDistance < distanceThreshold)
		{
			xAverageMovement += xDisplacement;
			yAverageMovement += yDisplacement;
			xAverageCounter++;
			yAverageCounter++;
			xLearnedMovement = (xAverageMovement / xAverageCounter);
			yLearnedMovement = (yAverageMovement / yAverageCounter);

			if(i > bufferMemory + 15)
				checkBasicXYAnomaly(xDisplacement, yDisplacement, tmpDetectPoint);
		}
	}
}


void individualTracking()
{
	const double distanceThreshold = 25;

	if(i == bufferMemory + 11 || ((carCoordinates.size() == 0) && i > bufferMemory + 10))
	{
 		registerFirstCar();
	}

	else if(detectedCoordinates.size() > 0)
	{

		vectorOfDetectedCars.push_back(detectedCoordinates);

		analyzeMovement();

		/*
		cout << "SECOND" << endl;

		for(int v = 0; v < carCoordinates.size(); v++)
		{
			cout << " DETECTED " << detectedCoordinates.size() << endl;
			cout << " CAR COORDINATES 2 " << carCoordinates.size() << endl;
			carCoordinates[v] = sortCoordinates(carCoordinates[v]);

			detectedCoordinates = sortCoordinates(detectedCoordinates);

			double lowestDistance = INT_MAX;
			double distance;

			Point tmpPoint;
			Point tmpDetectPoint;
			Point bestPoint;

			for(int j = 0; j < detectedCoordinates.size(); j++ )
			{
				tmpDetectPoint = detectedCoordinates[j];
				tmpPoint = carCoordinates[v][carCoordinates[v].size()-1];

				distance = sqrt(abs(tmpDetectPoint.x - tmpPoint.x) * (abs(tmpDetectPoint.y - tmpPoint.y)));

				cout << " ALL DISTANCES " << distance << endl;

				if(distance < lowestDistance)
				{
					lowestDistance = distance;
					bestPoint  = tmpDetectPoint;
					cout << " CAR DISTANCE IS " << distance << endl;
				}
				//carCoordinates.push_back();
			}

			int xDisplacement = abs(bestPoint.x - tmpPoint.x);
			int yDisplacement = abs(bestPoint.y - tmpPoint.y);

			cout << " LOWEST CAR DISTANCE IS " << lowestDistance << endl;
			cout << " X DISPLACEMENT is " <<xDisplacement<< endl;

			if(lowestDistance < distanceThreshold)
			{
				xAverageMovement += xDisplacement;
				yAverageMovement += yDisplacement;
				xAverageCounter++;
				yAverageCounter++;
				xLearnedMovement = (xAverageMovement / xAverageCounter);
				yLearnedMovement = (yAverageMovement / yAverageCounter);

				cout << " Y DISPLACEMENT is " <<yDisplacement << endl;

				cout << " AVERAGE X DISPLACEMENT IS " << xLearnedMovement << endl;
				cout << " AVERAGE Y DISPLACEMENT IS " << yLearnedMovement << endl;

				if(i > bufferMemory + 15)
					checkBasicXYAnomaly(xDisplacement, yDisplacement, tmpDetectPoint);
			}
		}
		*/
	}
}

void trackingML()
{
	if(i > bufferMemory + 10)
	{
		cout << " Entered ML" << endl;

		//displayCoordinates(detectedCoordinates);

		drawCoordinates(detectedCoordinates, "1st Pass");

 		//displayCoordinates(detectedCoordinates);

		//imwrite("trackingFrame.TIFF", trackingFrame);
 		detectedCoordinates = averageCoordinates(detectedCoordinates, 50);

 		displayCoordinates(detectedCoordinates);

		drawCoordinates(detectedCoordinates, "2nd");

  		individualTracking();

		drawAllTracking();

		cout << " Exited ML" << endl;
	}

	detectedCoordinates.erase(detectedCoordinates.begin(), detectedCoordinates.end());
}

//method to initalize Mats on startup
void initilizeMat()
{
	//if first run
	if(i == 0)
	{
		//initialize background subtractor object
		backgroundSubtractorGMM->set("initializationFrames", bufferMemory);
		backgroundSubtractorGMM->set("decisionThreshold", 0.85);

		//save gray value to set Mat parameters
		globalGrayFrames[i].copyTo(backgroundFrameMedian);

		/*
		globalFrames[i].copyTo(trackingFrame);

		for(int v = 0; v < trackingFrame.rows; v++)
			for(int j = 0; j < trackingFrame.cols; j++)
				trackingFrame.at<uchar>(v,j) = 0;
		*/
	}
}

//method to process exit of software
bool processExit(VideoCapture capture, clock_t t1, char keyboardClick)
{
	//if escape key is pressed
	if(keyboardClick==27)
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

//while buffering
void buffer()
{
}

//main method
int main() {

	//display welcome message if production code
	if(!debug)
		welcome();

	//creating initial and final clock objects
	//taking current time when run starts
	clock_t t1=clock();

	//random number generator
	RNG rng(12345);

	//defining VideoCapture object and filename to capture from
	VideoCapture capture(filename);

	//collecting statistics about the video
	//constants that will not change
	const int NUMBER_OF_FRAMES =(int) capture.get(CV_CAP_PROP_FRAME_COUNT);
	FRAME_RATE = (int) capture.get(CV_CAP_PROP_FPS);
	FRAME_WIDTH = capture.get(CV_CAP_PROP_FRAME_WIDTH);
	FRAME_HEIGHT = capture.get(CV_CAP_PROP_FRAME_HEIGHT);

	writeInitialStats(NUMBER_OF_FRAMES, FRAME_RATE, FRAME_WIDTH, FRAME_HEIGHT, filename);

	// declaring and initially setting variables that will be actively updated during runtime
	int framesRead = (int) capture.get(CV_CAP_PROP_POS_FRAMES);
	double framesTimeLeft = (capture.get(CV_CAP_PROP_POS_MSEC)) / 1000;

	//creating placeholder object
	Mat placeHolder = Mat::eye(1, 1, CV_64F);

	//vector to store execution times
	vector <string> FPS;

	//string to display execution time
	string strActiveTimeDifference;

	//actual run time, while video is not finished
	while(framesRead < NUMBER_OF_FRAMES)
	{
		clock_t tStart = clock();

		//read in current key press
		//char keyboardClick = cvWaitKey(33);

		//create pointer to new object
		Mat * frameToBeDisplayed = new Mat();

		//creating pointer to new object
		Mat * tmpGrayScale = new Mat();

		//reading in current frame
		capture.read(*frameToBeDisplayed);

		//for initial buffer read
		while(i < bufferMemory)
		{
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

			buffer();

			//display buffer progress
			if(!debug)
				cout << "Buffering frame " << i << ", " << (bufferMemory - i) << " frames remaining." << endl;

			//display splash screen
			welcome();

			//incrementing global counter
			i++;
		}

		//display splash screen
		welcome();

		//adding current frame to vector/array list of matricies
		globalFrames.push_back(*frameToBeDisplayed);
		Mat dispFrame;
		globalFrames[i].copyTo(dispFrame);
		putText(dispFrame,to_string(i),Point(0,50),3,1,Scalar(0,255,0),2);

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
		strActiveTimeDifference = (to_string(calculateFPS(tStart, tFinal))).substr(0, 4);

		//display performance
		if(debug)
			cout << "FPS is " << (to_string(1/(calculateFPS(tStart, tFinal)))).substr(0, 4) << endl;

		//saving FPS values
		FPS.push_back(strActiveTimeDifference);
		
		//running computer vision
		objectDetection(FRAME_RATE); 

		trackingML();

		//display frame number
		cout << "Currently processing frame number " << i << "." << endl;

		//method to process exit
		//if(processExit(capture,  t1, keyboardClick))
			//return 0;

		//deleting current frame from RAM
   		delete frameToBeDisplayed;

   		//incrementing global counter
   		i++;
	}

	//delete entire vector
   	globalFrames.erase(globalFrames.begin(), globalFrames.end());

	//compute run time
	computeRunTime(t1, clock(),(int) capture.get(CV_CAP_PROP_POS_FRAMES));

	//display finished, promt to close program
	cout << "Execution finished, file written, click to close window. " << endl;

	//wait for button press to proceed
	waitKey(0);

	//return code is finished and ran successfully
	return 0;
}
