//======================================================================================================
// Name        : TrafficCameraDistractedDriverDetection.cpp
// Author      : Vidur Prasad
// Version     : 0.3.0
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

CvHaarClassifierCascade *cascade;
CvMemStorage  *storage;

//vibe constructors
bgfg_vibe bgfg;
BackgroundSubtractorMOG bckSubMOG; // (200,  1, .7, 15);
 
//global frame properties
int FRAME_HEIGHT;
int FRAME_WIDTH;

//global counter
int i = 0;

//global completion variables for multithreading
int medianImageCompletion = 0;
int medianColorImageCompletion = 0;
int opticalFlowThreadCompletion = 0;
int opticalFlowAnalysisObjectDetectionThreadCompletion = 0;
int gaussianMixtureModelCompletion = 0;

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
 
//matrix storing GMM canny
Mat cannyGMM;
 
//matrix storing OFA thresh operations
Mat ofaThreshFrame;

//Mat objects to hold background frames
Mat backgroundFrameMedian;

//Mat for color background frame
Mat backgroundFrameColorMedian;
 
//Mat to hold temp GMM models
Mat gmmFrameRaw, binaryGMMFrame, gmmTempSegmentFrame;

//Mat for optical flow
Mat flow;
Mat cflow;
Mat optFlow;

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
const char* filename = "assets/froggerHighwayTCheck.mp4";

//defining format of data sent to threads 
struct thread_data{
   //include int for data passing
   int data;
};	

//function prototypes
Mat slidingWindowDetector(Mat srcFrame, int numRowSections, int numColumnSections);
Mat slidingWindowNeighborDetector(Mat srcFrame, int numRowSections, int numColumnSections);
Mat cannyContourDetector(Mat srcFrame);

//method to display frame
void displayFrame(string filename, Mat matToDisplay)
{
	//if in debug mode and Mat is not empty
	if(debug && matToDisplay.size[0] != 0)
	{imshow(filename, matToDisplay);}
}

//method to display frame overriding debug
void displayFrame(string filename, Mat matToDisplay, bool override)
{
	//if override and Mat is not empty
	if(override && matToDisplay.size[0] != 0){namedWindow(filename); imshow(filename, matToDisplay);}
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

	//deep copy grayscale frame
	globalGrayFrames.at(i-1).copyTo(thresholdFrameOFA);

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
				thresholdFrameOFA.at<uchar>(j,a) = 255;
			}
			else
			{
				//write to binary image
				thresholdFrameOFA.at<uchar>(j,a) = 0;
			}
		}
	}

	//performing sWND
	displayFrame("OFAOBJ pre" , thresholdFrameOFA );
	thresholdFrameOFA = slidingWindowNeighborDetector(thresholdFrameOFA, thresholdFrameOFA.rows / 10, thresholdFrameOFA.cols / 20);
	displayFrame("sWNDFrame1" , thresholdFrameOFA );
	thresholdFrameOFA = slidingWindowNeighborDetector(thresholdFrameOFA, thresholdFrameOFA.rows / 20, thresholdFrameOFA.cols / 40);
	displayFrame("sWNDFrame2" , thresholdFrameOFA );
	thresholdFrameOFA = slidingWindowNeighborDetector(thresholdFrameOFA, thresholdFrameOFA.rows / 30, thresholdFrameOFA.cols / 60);
	displayFrame("sWNDFrame3" , thresholdFrameOFA );
	thresholdFrameOFA = cannyContourDetector(thresholdFrameOFA);
	displayFrame("sWNDFrameCanny" , thresholdFrameOFA );

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

	//wait for completion
	while(opticalFlowAnalysisObjectDetectionThreadCompletion == 0) {}

	//reset completion variable
	opticalFlowAnalysisObjectDetectionThreadCompletion = 0;
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

	//signal thread completion
	opticalFlowThreadCompletion = 1;
	
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
		cout << to_string(i) << endl;
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
	   if(debug)
		   cout << ((activeCounter / displayPercentageCounter) * 100) << "% Median Image Scanned" << endl;

	}

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
	thresholdFrame = morph(thresholdFrame, 1, "closing");

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
	tmpStore = thresholdFrame(tmpStore, 8);
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
		cout << "Entered gray sale median" << endl;

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

		//perform Canny
		Mat gmmFrameSWNDCanny = cannyContourDetector(gmmFrame);
		displayFrame("CannyGMM", gmmFrameSWNDCanny);

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
	if(readMedianImg || !useMedians)
	{
		//read median image
		backgroundFrameMedian = imread("medianIMG.jpg");

		//convert to grayscale
		cvtColor( backgroundFrameMedian, backgroundFrameMedian, CV_BGR2GRAY );
 	}

	//if real-time calculation
	else
	{
		//after initial buffer read and using medians
		if(i == bufferMemory && useMedians)
		{ 
 			grayScaleFrameMedian();
		}
		//every 3 minutes
		if (i % (FRAME_RATE * 180) == 0 && i > 0)
		{
			//calculate new medians
 			grayScaleFrameMedian();
		}
	}
}

//method to draw canny contours
Mat cannyContourDetector(Mat srcFrame)
{
	//threshold for non-car objects or noise
	const int thresholdNoiseSize = 200;

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
  		if(arcLength(contours[v], true) > thresholdNoiseSize)
 		{
 			//draw object and circle center point
			drawContours( drawing, contours, v, Scalar(254,254,0), 2, 8, hierarchy, 0, Point() );
			circle( drawing, mc[v], 4, Scalar(254, 254, 0), -1, 8, 0 );
 		}
 	}

 	//return image with contours
	return drawing;
}

//method to do background subtraction with MOG 1
Mat bgMog(bool buffer)
{

	//instantiating Mat objects
	Mat fgmask;
	Mat bck;
	Mat fgMaskSWNDCanny;

	//performing background subtraction
    bckSubMOG.operator()(globalFrames.at(i), fgmask, .01); //1.0 / 200);
	
	if(!buffer)
	{ 
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
	}

	//return canny
	return fgMaskSWNDCanny;
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

//method to handle median image subtraction
Mat medianImageSubtraction(int FRAME_RATE)
{
	//generate or read background image
	generateBackgroundImage(FRAME_RATE);

	//calculate image difference and return
	return imageSubtraction();
}

//method to handle all image processing object detection
void objectDetection(int FRAME_RATE)
{
	//save all methods vehicle canny outputs
	Mat vibeDetection =	vibeBackgroundSubtraction(false);
	Mat mogDetection1 = bgMog(false);
	Mat mogDetection2 = bgMog2(false);
	Mat gmmDetection = gaussianMixtureModel();
	Mat medianDetection = medianImageSubtraction(FRAME_RATE);
	Mat ofaDetection = opticalFlowFarneback();

	//override display all frames
  	displayFrame("vibeDetection", vibeDetection, true);
  	displayFrame("mogDetection1", mogDetection1, true); 
 	displayFrame("mogDetection2", mogDetection2, true); 
 	displayFrame("gmmDetection", gmmDetection, true); 
 	displayFrame("medianDetection", medianDetection, true); 
 	displayFrame("ofaDetection", ofaDetection, true); 
 	displayFrame("Raw Frame", globalFrames[i], true);
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
		backgroundFrameMedian = globalGrayFrames[i];
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
	bgMog(true);
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
	const int FRAME_RATE = (int) capture.get(CV_CAP_PROP_FPS);
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
		if(!debug)
			cout << "FPS is " << (to_string(1/(calculateFPS(tStart, tFinal)))).substr(0, 4) << endl;

		//saving FPS values
		FPS.push_back(strActiveTimeDifference);
		
		//running computer vision
		objectDetection(FRAME_RATE); 

		//display frame number
		if(!debug)
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
