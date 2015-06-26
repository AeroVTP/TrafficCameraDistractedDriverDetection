//======================================================================================================
// Name        : TrafficCameraDistractedDriverDetection.cpp
// Author      : Vidur Prasad
// Version     : 0.2.4
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

////global variables////

//multithreading global variables
vector <Mat> globalFrames;
vector <Mat> globalGrayFrames;

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

//gaussian mixture model 
Ptr<BackgroundSubtractorGMG> backgroundSubtractorGMM = Algorithm::create<BackgroundSubtractorGMG>("BackgroundSubtractor.GMG");
bool firstTimeGMMModel = true;
Mat gmmFrame;

//optical flow density
int opticalFlowDensityDisplay = 10;

//Mat objects to hold background frames
Mat backgroundFrameMedian;

//Mat for color background frame
Mat backgroundFrameColorMedian;

//Mat subtracted image
Mat subtractedImage;

//Mat for thresholded binary image
Mat binaryFrame;

//Mat for optical flow
Mat flow;
Mat cflow;
Mat optFlow;

//Buffer memory size
const int bufferMemory = 300;

//boolean to decide if preprocessed median should be used
bool readMedianImg = false;

//controls all cout statements
bool debug = true;

//setting constant filename to read form
const char* filename = "assets/testRecordingSystemTCD3TCheck.mp4";
//const char* filename = "assets/ElginHighWayTCheck.mp4";

//defining format of data sent to threads 
struct thread_data{
   //include int for data passing
   int data;
};	

//method to display frame
void displayFrame(string filename, Mat matToDisplay)
{
	//if debug mode enabled display frame
	if(debug){imshow(filename, matToDisplay);}
}

//method to display frame overriding debug
void displayFrame(string filename, Mat matToDisplay, bool override)
{
	//if debug mode enabled display frame
	if(override){imshow(filename, matToDisplay);}
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
}

void *computeOpticalFlowAnalysisObjectDetection(void *threadarg)
{
	//reading in data sent to thread into local variable
	struct opticalFlowThreadData *data;
	data = (struct opticalFlowThreadData *) threadarg;

	//matrix holding temporary frame after threshold
	Mat thresholdFrame;

	//deep copy grayscale frame
	globalGrayFrames.at(i-1).copyTo(thresholdFrame);

	//set threshold
	const int threshold = 10;

	//iterating through OFA pixels
	for(int j = 0; j < cflow.rows; j++)
	{
		for (int a = 0 ; a < cflow.cols; a++)
		{
			const Point2f& fxy = flow.at<Point2f>(j, a);

			//if movement is greater than threshold
			if(sqrt((abs(fxy.x) * abs(fxy.y))) > threshold)
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

	//signal thread completion
   	opticalFlowAnalysisObjectDetectionThreadCompletion = 1;
}

//method to perform OFA threshold on Mat
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
		displayFrame("Gauss Frame", blurredFrame);

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

	//reading in current and previous frames
	prevFrame = blurFrame("gaussian", globalFrames.at(i-1), 15);

	currFrame = blurFrame("gaussian", globalFrames.at(i), 15);

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

void opticalFlowFarneback()
{
	//instantiate thread object
	pthread_t opticalFlowFarneback;

	//instantiating multithread Data object
	struct thread_data threadData;

	//saving data to pass
	threadData.data = i;

	//create OFA thread
	pthread_create(&opticalFlowFarneback, NULL, computeOpticalFlowAnalysisThread, (void *)&threadData);

	//wait till finished
	while(opticalFlowThreadCompletion == 0){}
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
	//display welcome image
	imshow("Welcome", imread("assets/IDCAST.jpg"));

	//put thread to sleep until user is ready
	this_thread::sleep_for (std::chrono::seconds(5));

	imshow("Welcome", imread("assets/TCD3Text.png"));

	//put thread to sleep until user is ready
	this_thread::sleep_for (std::chrono::seconds(7));

	//close welcome image
	destroyWindow("Welcome");
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

void *calcMedianImage(void *threadarg)
{
	//defining data structure to read in info to new thread
	struct thread_data *data;
	data = (struct thread_data *) threadarg;

	//reading in current iteration number
	int asdf = data->data;

	//performing deep copy
	globalGrayFrames.at(i).copyTo(backgroundFrameMedian);

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

//method to apply morphology
Mat morph(Mat sourceFrame, int amplitude, string type)
{
	//using default values
	int morph_elem = 0;
	int morph_size = 0;

	//constructing manipulation Mat
	Mat element = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );

	//if performing morphological closing
	if(type == "closing")
	{
		//repeat for increased effect
	    for(int v = 0; v < amplitude; v++)
		{
			 morphologyEx(sourceFrame, sourceFrame, MORPH_CLOSE, element,
					 Point(-1,-1), 20, BORDER_CONSTANT, morphologyDefaultBorderValue());
		}
	}

	//if performing morphological opening
	else if(type == "opening")
	{
		for(int v = 0; v < amplitude; v++)
		{
			//repeat for increased effect
			morphologyEx(sourceFrame, sourceFrame, MORPH_OPEN, element,
					Point(-1,-1), 20, BORDER_CONSTANT, morphologyDefaultBorderValue());

		}
	}

	//if performing morphological gradient
	else if(type == "gradient")
	{
		//repeat for increased effect
		for(int v = 0; v < amplitude; v++)
		{
			 morphologyEx(sourceFrame, sourceFrame, MORPH_GRADIENT, element,
					 Point(-1,-1), 20, BORDER_CONSTANT, morphologyDefaultBorderValue());
		}
	}

	//if performing morphological tophat
	else if(type == "tophat")
	{
		//repeat for increased effect
		for(int v = 0; v < amplitude; v++)
		{
			 morphologyEx(sourceFrame, sourceFrame, MORPH_TOPHAT, element,
					 Point(-1,-1), 20, BORDER_CONSTANT, morphologyDefaultBorderValue());
		}
	}

	//if performing morphological blackhat
	else if(type == "blackhat")
	{
		//repeat for increased effect
		for(int v = 0; v < amplitude; v++)
		{
		morphologyEx(sourceFrame, sourceFrame, MORPH_BLACKHAT, element,
							 Point(-1,-1), 20, BORDER_CONSTANT, morphologyDefaultBorderValue());
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

//method to threshold standard frame
Mat thresholdFrame(Mat sourceDiffFrame)
{
	//Mat to hold frame
	Mat thresholdFrame;

	//perform deep copy into destination Mat
	sourceDiffFrame.copyTo(thresholdFrame);

	//threshold value
	const int threshold = 25;

	//steping through pixels
	for(int j=0;j<sourceDiffFrame.rows;j++)
	{
	    for (int a=0;a<sourceDiffFrame.cols;a++)
	    {
	    	//if pixel value greater than threshold
	    	if(sourceDiffFrame.at<uchar>(j,a) > threshold)
	    	{
	    		//write to binary image
	    		thresholdFrame.at<uchar>(j,a) = 0;
	    	}
	    	else
	    	{
	    		//write to binary image
	    		thresholdFrame.at<uchar>(j,a) = 255;
	    	}
	    }
	}

	//perform morphology
	thresholdFrame = morph(thresholdFrame, 1000000, "closing");

	//display frame
	displayFrame("SBckSub Bin Frame", thresholdFrame);

	//return thresholded frame
	return thresholdFrame;
}

//method to perform simple image subtraction
Mat imageSubtraction()
{
	//subtract, perform blur, threshold, apply morphology, and return
	return thresholdFrame((blurFrame("gaussian", (globalGrayFrames.at(i) - backgroundFrameMedian), 5)));
}

//method to perform median on grayscale images
void grayScaleFrameMedian()
{
	//instantiating multithread object
	pthread_t medianImageThread;

	//instantiating multithread Data object
	struct thread_data threadData;

	//saving data into multithread
	threadData.data = i;

	//creating thread to calculate median of image
	pthread_create(&medianImageThread, NULL, calcMedianImage, (void *)&threadData);

	//wait for completion
	while(medianImageCompletion != 1){}

	//reset completion variable
	medianImageCompletion = 0;

	//display grayscale median
	displayFrame("GrayScale Median Background Image", backgroundFrameMedian);

	//save median image
	imwrite((currentDateTime() + "medianBackgroundImage.jpg"), backgroundFrameMedian);
}

//method to calculate median of color image
void *calcMedianColorImage(void *threadarg)
{
	//defining data structure to read in info to new thread
	struct thread_data *data;
	data = (struct thread_data *) threadarg;

	//performing deep copy
	globalFrames.at(i).copyTo(backgroundFrameColorMedian);

	//variables to show percentage complete
	double displayPercentageCounter = 0;
	double activeCounter = 0;

	//calculating number of runs
	for(int j=0;j<backgroundFrameColorMedian.rows;j++)
	{
		for (int a=0;a<backgroundFrameColorMedian.cols;a++)
		{
			for (int t = (i - bufferMemory); t < i ; t++)
			{
				displayPercentageCounter++;
			}
		}
	}

	//iterate through pixels
	for(int j=0;j<backgroundFrameMedian.rows;j++)
	{
		for (int a=0;a<backgroundFrameMedian.cols;a++)
		{
			//vectors to hold BGR pixel values
			vector <int> pixelHistoryBlue;
			vector <int> pixelHistoryGreen;
			vector <int> pixelHistoryRed;

			//iterating through buffer
			for (int t = (i - bufferMemory); t < i ; t++)
			{
				//save pixel value into 3D vector
				Vec3b bgrPixel = globalFrames.at(i).at<Vec3b>(j, a);

				//create matrix to save frame to
				Mat currentFrameForMedianBackground;

				//perform deep copy of frame
				globalFrames.at(i-t).copyTo(currentFrameForMedianBackground);

				//save BGR pixel values
				pixelHistoryBlue.push_back((int) backgroundFrameColorMedian.at<cv::Vec3b>(j,a)[0]);
				pixelHistoryGreen.push_back((int) backgroundFrameColorMedian.at<cv::Vec3b>(j,a)[1]);
				pixelHistoryRed.push_back((int) backgroundFrameColorMedian.at<cv::Vec3b>(j,a)[2]);

				//increment progress counter
				activeCounter++;
			}

			//calculate medians and save in image
			backgroundFrameColorMedian.at<Vec3b>(j,a)[0] = calcMedian(pixelHistoryBlue);
			backgroundFrameColorMedian.at<Vec3b>(j,a)[1] = calcMedian(pixelHistoryGreen);
			backgroundFrameColorMedian.at<Vec3b>(j,a)[2] = calcMedian(pixelHistoryRed);

	   }

	   //display completion stats
	   if(debug)
		   cout << ((activeCounter / displayPercentageCounter) * 100) << "% Color Median Image Scanned" << endl;

	}

	//display color median
	displayFrame("Background Frame Color Median", backgroundFrameColorMedian);

	//signal completion
	medianColorImageCompletion = 1;
}

//method to calculate Gaussian image difference
void *calcGaussianMixtureModel(void *threadarg)
{
	//Mat to hold temp GMM models
    Mat gmmFrameRaw, binaryGMMFrame, gmmTempSegmentFrame;

    //if first run through scan all images
	if(firstTimeGMMModel)
	{
		//initialize background subtractor object
		backgroundSubtractorGMM->set("initializationFrames", 20);
		backgroundSubtractorGMM->set("decisionThreshold", 0.7);

		//step through all frames
		for(int v = 0; v < i; v++)
		{
			//perform deep copy
			globalGrayFrames.at(v).copyTo(gmmFrameRaw);

			//perform GMM
			(*backgroundSubtractorGMM)(gmmFrameRaw, binaryGMMFrame);

			//save into tmp frame
			gmmFrameRaw.copyTo(gmmTempSegmentFrame);

			//add movement mask
			add(gmmFrameRaw, Scalar(0, 255, 0), gmmTempSegmentFrame, binaryGMMFrame);

			//perform morphology
			binaryGMMFrame = morph(binaryGMMFrame, 1000000, "closing");

			//save into display file
			gmmFrame = binaryGMMFrame;
		}
	}

	//if run time
	else
	{
		//perform deep copy
		globalGrayFrames.at(i).copyTo(gmmFrameRaw);

		//update model
		(*backgroundSubtractorGMM)(gmmFrameRaw, binaryGMMFrame);

		//save into tmp frame
		gmmFrameRaw.copyTo(gmmTempSegmentFrame);

		//add movement mask
		add(gmmFrameRaw, Scalar(0, 255, 0), gmmTempSegmentFrame, binaryGMMFrame);

		//perform morphology
		binaryGMMFrame = morph(binaryGMMFrame, 1000000, "closing");

		//save into display file
		gmmFrame = binaryGMMFrame;
	}

	//signal thread completion
	gaussianMixtureModelCompletion = 1;
}

void blobDetector()
{
	// Set up the detector with default parameters.
	SimpleBlobDetector detector;

	// Detect blobs.
	std::vector<KeyPoint> keypoints;
	detector.detect( gmmFrame, keypoints);

	// Draw detected blobs as red circles.
	Mat im_with_keypoints;
	drawKeypoints( gmmFrame, keypoints, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

	// Show blobs
	imshow("keypoints", im_with_keypoints );
}

//method to handle GMM thread
void gaussianMixtureModel()
{
	//instantiate thread object
	pthread_t gaussianMixtureModelThread;

	//instantiating multithread Data object
	struct thread_data threadData;

	//save i data
	threadData.data = i;

	//create thread
	pthread_create(&gaussianMixtureModelThread, NULL, calcGaussianMixtureModel, (void *)&threadData);

	//wait for finish
	while(gaussianMixtureModelCompletion != 1){}

	//reset completion variable
	gaussianMixtureModelCompletion = 0;

	//display frame
	displayFrame("GMM Frame", gmmFrame);

	//flag first time run complete
	firstTimeGMMModel = false;
}

//method to handle color frame median
void colorFrameMedian()
{
	//instantiate thread object
	pthread_t medianImageColorThread;

	//instantiating multithread Data object
	struct thread_data threadData;

	//save i data
	threadData.data = i;

	//create thread
	pthread_create(&medianImageColorThread, NULL, calcMedianColorImage, (void *)&threadData);

	//wait for finish
	while(medianColorImageCompletion != 1)	{}

	//reset completion variable
	medianColorImageCompletion = 0;
	
	//display colormedian image
	displayFrame("Median Color Background Image", backgroundFrameColorMedian);

	//write to file
	imwrite((currentDateTime() + "medianColorBackgroundImage.jpg"), backgroundFrameMedian);
}

//method to handle all background image generation
void generateBackgroundImage(int FRAME_RATE)
{
	//if post-processing
	if(readMedianImg)
	{
		backgroundFrameMedian = imread("/Users/Vidur/OneDrive/EX/Internships/ID\ CAST/Traffic\ Camera\ Distracted\ Driver\
				Detection/C\ Workspace/Traffic\ Camera\ Distracted\ Driver\ Detection/2015-06-24.12\:29\:43medianBackgroundImage.jpg");
	}

	//if real-time calculation
	else
	{
		//after initial buffer read
		if(i == bufferMemory)
		{
			//performing initialGMM
			gaussianMixtureModel();

			//calculating medians
			colorFrameMedian();
			grayScaleFrameMedian();
		}
		//every minute
		if (i % (FRAME_RATE * 60) == 0 && i > 0)
		{
			//calculate new medians
			colorFrameMedian();
			grayScaleFrameMedian();
		}
	}
}

//method to handle all image processing object detection
void objectDetection(int FRAME_RATE)
{
	//updating gaussian model
	gaussianMixtureModel();

	//create background image
	generateBackgroundImage(FRAME_RATE);

	//perform OFA for motion object detection
	opticalFlowFarneback();

	//perform simple image subtraction
	imageSubtraction();
}

//method to initalize Mats on startup
void initilizeMat()
{
	//if first run
	if(i == 0)
	{
		//save gray value to set Mat parameters
		backgroundFrameMedian = globalGrayFrames.at(i);
	}
}

bool processExit(VideoCapture capture, clock_t t1)
{
	//read in current key press
	char c = cvWaitKey(33);


	//if escape key is pressed
	if(c==27)
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
			cvtColor(globalFrames.at(i), *tmpGrayScale, CV_BGR2GRAY);

			//save grayscale frame
			globalGrayFrames.push_back(*tmpGrayScale);

			//initilize Mat objects
			initilizeMat();

			//display buffer progress
			if(debug)
				cout << "Buffering frame " << i << ", " << (bufferMemory - i) << " frames remaining." << endl;

			//incrementing global counter
			i++;
		}

		//adding current frame to vector/array list of matricies
		globalFrames.push_back(*frameToBeDisplayed);

		//display raw frame
		displayFrame("RCFrame", globalFrames.at(i));

		//convert to gray scale
		cvtColor(globalFrames.at(i), *tmpGrayScale, CV_BGR2GRAY);

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
		cout << (to_string(1/(calculateFPS(tStart, tFinal)))).substr(0, 4) << endl;

		//saving FPS values
		FPS.push_back(strActiveTimeDifference);
		
		//running image analysis
		objectDetection(FRAME_RATE);

		//display frame number
		if(debug)
			{cout << "Currently processing frame number " << i << "." << endl;}

		//method to process exit
		if(processExit(capture,  t1))
			return 0;

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
