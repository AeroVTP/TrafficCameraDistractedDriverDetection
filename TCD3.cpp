//======================================================================================================
// Name        : TrafficCameraDistractedDriverDetection.cpp
// Author      : Vidur Prasad
// Version     : 0.2.2
// Copyright   : Institute for the Development and Commercialization of Advanced Sensor Technology Inc.
// Description : Detect Drunk, Distracted, and Anomlous Driving Using Traffic Cameras
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
vector <Mat> globalMedianGrayFrames;

int FRAME_HEIGHT;
int FRAME_WIDTH;

//global counter
int i = 0;

//global variables for multithreading
int medianImageCompletion = 0;
int opticalFlowThreadCompletion = 0;
int opticalFlowAnalysisObjectDetectionThreadCompletion = 0;

int opticalFlowDensityDisplay = 5;

//Mat objects to hold background frames
Mat backgroundFrameMedian;

//Mat subtracted image
Mat subtractedImage;

//Mat for thresholded binary image
Mat binaryFrame;

Mat optFlow;

Mat opticalFlowObject;

Mat flow;
Mat cflow;

//Median memory
const int medianMemory = 25;

//boolean to decide if preprocessed median should be used
bool readMedianImg = true;

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
struct opticalFlowThreadData
{
	Mat cflowmap;
	Mat flow;
};

void displayFrame(string filename, Mat matToDisplay)
{
	if(debug){imshow(filename, matToDisplay);}
}

//method to draw optical flow, only should be called during demos
static void drawOptFlowMap(const Mat& flow, Mat& cflowmap,
                    double, const Scalar& color)
{
	if(debug){} //cout << "entering drawOptFlowMap" << endl;}

	//iterating through each pixel and drawing vector
    for(int y = 0; y < cflowmap.rows; y += opticalFlowDensityDisplay)
    {
    	for(int x = 0; x < cflowmap.cols; x += opticalFlowDensityDisplay)
        {
            const Point2f& fxy = flow.at<Point2f>(y, x);
            if(debug)
            	{} //cout << sqrt((abs(fxy.x) * abs(fxy.y))) << " size of x" << endl;}
            line(cflowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),
                 color);
            circle(cflowmap, Point(x,y), 0, color, -1);
        }
   	}

    displayFrame("OFA", cflowmap);
    if(debug){} //cout << "exiting drawOptFlowMap" << endl;
}

void *computeOpticalFlowAnalysisObjectDetection(void *threadarg)
{
	//reading in data sent to thread into local variable
	struct opticalFlowThreadData *data;
	data = (struct opticalFlowThreadData *) threadarg;

	Mat thresholdFrame;

	globalGrayFrames.at(i-1).copyTo(thresholdFrame);

	if(debug){} // {cout << "max value is " << maxMat(sourceDiffFrame) << endl; }

	const int threshold = 10;

	for(int j=0;j<cflow.rows;j++)
	{
		for (int a=0;a<cflow.cols;a++)
		{
			const Point2f& fxy = flow.at<Point2f>(j, a);

			if(sqrt((abs(fxy.x) * abs(fxy.y))) > threshold)
			{
				thresholdFrame.at<uchar>(j,a) = 0;
				if(!debug){}
					//thresholdFrame = drawObjectLocation(j,a, thresholdFrame);
			}
			else
			{
				thresholdFrame.at<uchar>(j,a) = 255;
			}
		}
	}

	/*
	for(int j=0;j<cflow.rows;j++)
	{
		for (int a=0;a<cflow.cols;a++)
		{
			if(cflow.at<uchar>(j,a) > threshold)
			{
				thresholdFrame.at<uchar>(j,a) = 0;
				if(!debug){}
					//thresholdFrame = drawObjectLocation(j,a, thresholdFrame);
			}
			else
			{
				thresholdFrame.at<uchar>(j,a) = 255;
			}
		}
	}
	*/
	displayFrame("OFA Object", thresholdFrame);



	/*
	for(int y = 0; y < cflow.rows; y += opticalFlowDensityDisplay)
	{
	   	for(int x = 0; x < cflow.cols; x += opticalFlowDensityDisplay)
	    {
	   		counterCheck++;
	    }
	}

	//iterating through each pixel and drawing vector
    for(int y = 0; y < opticalFlowObject.rows; y += opticalFlowDensityDisplay)
    {
    	for(int x = 0; x < opticalFlowObject.cols; x += opticalFlowDensityDisplay)
        {
    		liveCounter++;

    		if(debug)
    		{
    			//cout << liveCounter << " live counter" << endl;
    			//cout << counterCheck << " counter check" << endl;
    		}

    		if(debug){}
    			//cout << "entered here adsf " << endl;

    		const Point2f& fxy = flow.at<Point2f>(y, x);
            if(debug){}
            	//cout << sqrt((abs(fxy.x) * abs(fxy.y))) << " size of x" << endl;

            //opticalFlowObject.at<uchar>(x,y) = (uchar) sqrt((abs(fxy.x) * abs(fxy.y)));
        }
   	}
	*/

   	opticalFlowAnalysisObjectDetectionThreadCompletion = 1;
}

bool opticalFlowAnalysisObjectDetection(Mat& cflowmap, Mat& flow)
{
	pthread_t opticalFlowAnalysisObjectDetectionThread;

	//instantiating multithread Data object
	struct opticalFlowThreadData threadData;

	threadData.cflowmap = cflowmap;
	threadData.flow = flow;

	if(debug){}
		//cout << "asdf asdf" << endl;

	pthread_create(&opticalFlowAnalysisObjectDetectionThread, NULL, computeOpticalFlowAnalysisObjectDetection, (void *)&threadData);
	while(opticalFlowAnalysisObjectDetectionThreadCompletion == 0) {}
	opticalFlowAnalysisObjectDetectionThreadCompletion = 1;

	return true;

	if(debug){}
		//cout << " yay" << endl;

	if(opticalFlowAnalysisObjectDetectionThreadCompletion != 1)
	{
		return false;
	}

	else
	{
		opticalFlowAnalysisObjectDetectionThreadCompletion = 0;
		return true;
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
	prevFrame = globalFrames.at(i-1);
	currFrame = globalFrames.at(i);

	//converting to grayscale
	cvtColor(currFrame, gray,COLOR_BGR2GRAY);
	cvtColor(prevFrame, prevGray, COLOR_BGR2GRAY);

	//calculating optical flow
	calcOpticalFlowFarneback(prevGray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
	//converting to display format
	cvtColor(prevGray, cflow, COLOR_GRAY2BGR);

	bool complete = opticalFlowAnalysisObjectDetection(flow, cflow);
	if(debug)
	{
		//cout << "moving forward" << endl;
		//drawing optical flow vectors
		drawOptFlowMap(flow, cflow, 1.5, Scalar(0, 0, 255));
	}

	//saving to global variable for display
	optFlow = cflow;

	//displayFrame("Optical Flow", optFlow);

	while(!complete)
	{
		opticalFlowThreadCompletion = 0;
	}

	opticalFlowThreadCompletion = 1;
	/*
	if(complete)
	{
		//signal thread completion
		opticalFlowThreadCompletion = 1;
	}
	*/
	
	if(debug){}//cout << " qwerasdf" << endl;}

}

bool opticalFlowFarneback()
{
	pthread_t opticalFlowFarneback;

	//instantiating multithread Data object
	struct thread_data threadData;

	threadData.data = i;

	pthread_create(&opticalFlowFarneback, NULL, computeOpticalFlowAnalysisThread, (void *)&threadData);

	if(debug){}
		//cout << " finish " << endl;

	while(opticalFlowThreadCompletion == 0){}

	return true;
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

	//creating filename ending
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

	//close welcome image
	destroyWindow("Welcome");
}
//calculate time for each iteration
double calculateFPS(clock_t tStart, clock_t tFinal)
{
	//return frames per second
	return 1/((((float)tFinal-(float)tStart) / CLOCKS_PER_SEC));
}
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
double calcMedian(vector<int> integers)
{
  double median;

  size_t size = integers.size();

  sort(integers.begin(), integers.end());

  if (size % 2 == 0)
  {
      median = (integers[size / 2 - 1] + integers[size / 2]) / 2;
  }
  else
  {
      median = integers[size / 2];
  }
  return median;
}
string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

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

  r += "C";
  r += (chans+'0');

  return r;
}
double calcMean(vector <int> integers)
{
	int total = 0;
	for (int v = 0; v < integers.size(); v++)
	{
		total += integers.at(v);
	}

	return total/integers.size();
	//return sum(integers)[0] / integers.size();
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

	double displayPercentageCounter = 0;
	double activeCounter = 0;

	//calculating number of runs
	for(int j=0;j<backgroundFrameMedian.rows;j++)
	{
	    for (int a=0;a<backgroundFrameMedian.cols;a++)
	    {
		  	for (int t = (i - medianMemory); t < i ; t++)
		  	{
		  		displayPercentageCounter++;
		  	}
	    }
	}

	//change some pixel value
	for(int j=0;j<backgroundFrameMedian.rows;j++)
	{
	    for (int a=0;a<backgroundFrameMedian.cols;a++)
	    {
   			int totalCounter = 1;
		    vector <int> pixelHistory;
		  	for (int t = (i - medianMemory); t < i ; t++)
		    {

		    	Mat currentFrameForMedianBackground;
			    globalGrayFrames.at(i-t).copyTo(currentFrameForMedianBackground);
			    pixelHistory.push_back(currentFrameForMedianBackground.at<uchar>(j,a));
			    totalCounter++;
			    activeCounter++;
		    }
		  
			backgroundFrameMedian.at<uchar>(j,a) = calcMedian(pixelHistory);
	   }

	  if(debug)
		  cout << ((activeCounter / displayPercentageCounter) * 100) << "% Median Image Scanned" << endl;

	}
	putText(backgroundFrameMedian, to_string(i), cvPoint(30,30),CV_FONT_HERSHEY_SIMPLEX, 1, cvScalar(0,255,0), 1, CV_AA, false);

    medianImageCompletion = 1;
    
}
int maxMat(Mat sourceFrame)
{
	int currMax = INT_MIN;

	for(int j=0;j<sourceFrame.rows;j++)
	{
	    for (int a=0;a<sourceFrame.cols;a++)
	    {
	    	if(sourceFrame.at<uchar>(j,a) > currMax)
	    	{
	    		currMax = sourceFrame.at<uchar>(j,a);
	    	}
	    }
	}

	return currMax;
}
Mat drawObjectLocation(int j, int a, Mat sourceFrame)
{
	circle(sourceFrame, Point(j,a), 10, Scalar( 1, 1, 254), 1, 8, 0);
	return sourceFrame;
}

Mat morph(Mat sourceFrame, int amplitude, string type)
{
	if(type == "closing")
	{
		int morph_elem = 0;
		int morph_size = 0;
		int morph_operator = 0;
		int const max_operator = 4;
		int const max_elem = 2;
		int const max_kernel_size = 21;

		 Mat element = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );

		 for(int v = 0; v < amplitude; v++)
		 {
			 morphologyEx(sourceFrame, sourceFrame, MORPH_CLOSE, element,
					 Point(-1,-1), 20, BORDER_CONSTANT, morphologyDefaultBorderValue());
		 }

		 return sourceFrame;

		//morphologyEx(thresholdFrame, thresholdFrame, MORPH_CLOSE, element);
	}

	else
	{
		if(debug)
			cout << type <<  " type of morph not implemented yet" << endl;
	}
}

Mat thresholdFrame(Mat sourceDiffFrame)
{
	Mat thresholdFrame;

	sourceDiffFrame.copyTo(thresholdFrame);

	if(debug){} // {cout << "max value is " << maxMat(sourceDiffFrame) << endl; }

	const int threshold = 25;

	for(int j=0;j<sourceDiffFrame.rows;j++)
	{
	    for (int a=0;a<sourceDiffFrame.cols;a++)
	    {
	    	if(sourceDiffFrame.at<uchar>(j,a) > threshold)
	    	{
	    		thresholdFrame.at<uchar>(j,a) = 0;
	    		if(!debug)
	    			thresholdFrame = drawObjectLocation(j,a, thresholdFrame);
	    	}
	    	else
	    	{
	    		thresholdFrame.at<uchar>(j,a) = 255;
	    	}
	    }
	}

	thresholdFrame = morph(thresholdFrame, 1000000, "closing");

	displayFrame("Binary Frame", thresholdFrame);

	return thresholdFrame;
}
Mat blurFrame(string blurType, Mat sourceDiffFrame, int blurSize)
{
	Mat blurredFrame;
	if(blurType == "gaussian")
	{
		blur(sourceDiffFrame, blurredFrame, Size (blurSize,blurSize), Point(-1,-1));
		displayFrame("blurredFrame", blurredFrame);
		return blurredFrame;
	}

	else
		{
			if(debug)
				cout << blurType <<  " type of blur not implemented yet" << endl;
		}

}
Mat imageSubtraction()
{
	return thresholdFrame((blurFrame("gaussian", (globalGrayFrames.at(i) - backgroundFrameMedian), 5)));
}
void generateBackgroundImage(int FRAME_RATE)
{
	if(readMedianImg)
	{
		backgroundFrameMedian = imread("/Users/Vidur/OneDrive/EX/Internships/ID\ CAST/Traffic\ Camera\ Distracted\ Driver\
				Detection/C\ Workspace/Traffic\ Camera\ Distracted\ Driver\ Detection/2015-06-24.12\:29\:43medianBackgroundImage.jpg");
	}

	else
	{
		if ((i % (FRAME_RATE * 60) == 0 && i > 2) || i == 30)
		{
			pthread_t medianImageThread;

			//instantiating multithread Data object
			struct thread_data threadData;

			threadData.data = i;


			pthread_create(&medianImageThread, NULL, calcMedianImage, (void *)&threadData);
			while(medianImageCompletion != 1)
			{

			}
			//pthread_create(&meanImageThread, NULL, calcMeanImage, (void *)&threadData);

			//backgroundFrameMedian = meanImage();
			//backgroundFrameMedian = medianImage();
			//while(medianImageCompletion != 1 || medianImageCompletion !=  1) {}

			medianImageCompletion = 0;

			/*
			if(i % 10 == 0 && i > 9)
			{
				destroyWindow("Median Background Image");
				destroyWindow("Mean Background Image");
			}
			*/

			//imshow("Median Background Image", backgroundFrameMedian);
			//destroyWindow("Mean Background Image");
			imshow("Median Background Image", backgroundFrameMedian);
			imwrite((currentDateTime() + "medianBackgroundImage.jpg"), backgroundFrameMedian);

		}
	}
}
void displayWindows()
{
	Mat * dispMat = new Mat();

	*dispMat = globalFrames.at(i);

	putText(*dispMat, to_string(i), cvPoint(30,30),CV_FONT_HERSHEY_SIMPLEX, 1, cvScalar(0,255,0), 1, CV_AA, false);

	imshow("Raw Frame",*dispMat );

	imshow("Raw Car Image", subtractedImage);
}

void objectDetection(int FRAME_RATE)
{
	generateBackgroundImage(FRAME_RATE);

	opticalFlowFarneback();

	imageSubtraction();

	while(opticalFlowThreadCompletion == 0 && opticalFlowAnalysisObjectDetectionThreadCompletion == 0){ if(debug){cout << "waiting" << endl;}}

	opticalFlowThreadCompletion = 1;

	displayFrame("OFA Thresh", opticalFlowObject);

}

void initilizeMat()
{
	if(i == 0)
	{
		opticalFlowObject = globalGrayFrames.at(i);
		backgroundFrameMedian = globalGrayFrames.at(i);
	}
}

//main method
int main() {

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

	vector <string> FPS;
	string strActiveTimeDifference;

	Mat globalGrayFrame;

	//actual run time, while video is not finished
	while(framesRead < NUMBER_OF_FRAMES)
	{
		clock_t tStart = clock();

		//create pointer to new object
		Mat * frameToBeDisplayed = new Mat();

		//reading in current frame
		capture.read(*frameToBeDisplayed);


		while(i < medianMemory)
		{
			//reading in current frame
			capture.read(*frameToBeDisplayed);

			//adding current frame to vector/array list of matricies
			globalFrames.push_back(*frameToBeDisplayed);

			//globalFrames.at(i).convertTo(globalGrayFrames.at(i), CV_8U);
			Mat * tmpGrayScale = new Mat();

			cvtColor(globalFrames.at(i), *tmpGrayScale, CV_BGR2GRAY);
			globalGrayFrames.push_back(*tmpGrayScale);
			globalMedianGrayFrames.push_back(*tmpGrayScale);

			initilizeMat();

			destroyWindow("Buffer Frames");
			imshow("Buffer Frames", globalFrames.at(i));

			if(debug)
				cout << "Buffering frame " << i << ", " << (medianMemory - i) << " frames remaining." << endl;

			i++;
		}

		if( i == medianMemory)
		{
			destroyWindow("Buffer Frames");
		}

		//adding current frame to vector/array list of matricies
		globalFrames.push_back(*frameToBeDisplayed);

		//globalFrames.at(i).convertTo(globalGrayFrames.at(i), CV_8U);
		Mat * tmpGrayScale = new Mat();

		cvtColor(globalFrames.at(i), *tmpGrayScale, CV_BGR2GRAY);
		globalGrayFrames.push_back(*tmpGrayScale);
		globalMedianGrayFrames.push_back(*tmpGrayScale);

		if (i == 0)
		{
			backgroundFrameMedian = globalGrayFrames.at(i);
		}

		//globalGrayFrames.push_back(globalGrayFrame);

		//gather real time statistics
		framesRead = (int) capture.get(CV_CAP_PROP_POS_FRAMES);
		framesTimeLeft = (capture.get(CV_CAP_PROP_POS_MSEC)) / 1000;

		//clocking end of run time
		clock_t tFinal = clock();

		//calculate time
		strActiveTimeDifference = (to_string(calculateFPS(tStart, tFinal))).substr(0, 4);
		//cout <<(to_string(1/(calculateFPS(tStart, tFinal)))).substr(0, 4) << endl;
		//saving FPS values
		FPS.push_back(strActiveTimeDifference);
		
		objectDetection(FRAME_RATE);

		if(debug)
			{/*cout << i << endl;*/}

		//method to display frames
		//displayWindows(i);

		//read in current key press
		char c = cvWaitKey(33);

		//if escape key is pressed
		if(c==27)
		{
			//reset key listener
			c = cvWaitKey(33);

			//display warning
			cout << "Are you sure you want to exit?" << endl;

			//if key is pressed again, in rapid succession
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
				return 0;
			}
		}

		/*
   		//after keeping adequate buffer of 3 frames
   		if(i > 3)
   		{
   			//deleting current frame from RAM
   			delete frameToBeDisplayed;

   			//replacing old frames with low RAM placeholder
   			globalFrames.erase(globalFrames.begin() + (i - 3));
   			globalFrames.insert(globalFrames.begin(), placeHolder);
   		}
   		*/

   		//incrementing counter
   		i++;
	}

	//delete entire vector
   	globalFrames.erase(globalFrames.begin(), globalFrames.end());

	//normalize all ratings

	//compute run time
	computeRunTime(t1, clock(),(int) capture.get(CV_CAP_PROP_POS_FRAMES));

	//display finished, promt to close program
	cout << "Execution finished, file written, click to close window. " << endl;

	//wait for button press to proceed
	waitKey(0);

	//return code is finished and ran successfully
	return 0;
}
