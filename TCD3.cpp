//======================================================================================================
// Name        : TrafficCameraDistractedDriverDetection.cpp
// Author      : Vidur Prasad
// Version     : 0.1.0
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

//multithreading global variables
vector <Mat> globalFrames;
vector <Mat> globalGrayFrames;

int FRAME_HEIGHT;
int FRAME_WIDTH;

//global counter
int i = 0;

//setting constant filename to read form
const char* filename = "assets/testRecordingSystemTCD3TCheck.mp4";

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

	//display video statistics
	cout << "Stats on video >> There are = " << NUMBER_OF_FRAMES << " frames. The frame rate is " << FRAME_RATE
	<< " frames per second. Resolution is " << FRAME_WIDTH << " X " << FRAME_HEIGHT << endl;;

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
double calcMean(vector <int> integers)
{
	return sum(integers)[0] / integers.size();
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

Mat medianImage()
{
	Mat medianImage;

	medianImage = globalGrayFrames.at(i);

	Mat img = globalGrayFrames.at(i);
	//change some pixel value
	for(int j=0;j<img.rows;j++)
	{
	  for (int a=0;a<img.cols;a++)
	  {
		  vector <int> pixelHistory;
		  for (int t = 0; t < i ; t++)
		  {
			pixelHistory.push_back((globalGrayFrames.at(t)).at<uchar>(j,a));
		  }
		  medianImage.at<uchar>(j,a) = calcMedian(pixelHistory);



	  }
	  //cout << j << endl;
	}

	putText(medianImage, to_string(i), cvPoint(30,30),CV_FONT_HERSHEY_SIMPLEX, 1, cvScalar(0,255,0), 1, CV_AA, false);
	imshow("Median Image", medianImage);
	return medianImage;
}
Mat meanImage()
{
	Mat meanImage;

	meanImage = globalGrayFrames.at(i);

	Mat img = globalGrayFrames.at(i);
	//change some pixel value
	for(int j=0;j<img.rows;j++)
	{
	  for (int a=0;a<img.cols;a++)
	  {
		  vector <int> pixelHistory;
		  for (int t = 0; t < i ; t++)
		  {
			pixelHistory.push_back((globalGrayFrames.at(t)).at<uchar>(j,a));
		  }

		  meanImage.at<uchar>(j,a) = calcMean(pixelHistory);
		  //medianImage.at<uchar>(j,a) = rand() % 250;
		  //cout << a << "A" << endl;

	  }
	  //cout << j << endl;
	}
	putText(meanImage, to_string(i), cvPoint(30,30),CV_FONT_HERSHEY_SIMPLEX, 1, cvScalar(0,255,0), 1, CV_AA, false);
	imshow("Mean Image", meanImage);
	return meanImage;
}

//main method
int main() {

	//welcome();

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

	//writeInitialStats(NUMBER_OF_FRAMES, FRAME_RATE, FRAME_WIDTH, FRAME_HEIGHT, filename);

	// declaring and initially setting variables that will be actively updated during runtime
	int framesRead = (int) capture.get(CV_CAP_PROP_POS_FRAMES);
	double framesTimeLeft = (capture.get(CV_CAP_PROP_POS_MSEC)) / 1000;

	//creating placeholder object
	Mat placeHolder = Mat::eye(1, 1, CV_64F);

	vector <string> FPS;
	string strActiveTimeDifference;

	Mat backgroundFrame;
	Mat globalGrayFrame;

	//actual run time, while video is not finished
	while(framesRead < NUMBER_OF_FRAMES)
	{
		clock_t tStart = clock();

		//create pointer to new object
		Mat * frameToBeDisplayed = new Mat();

		//reading in current frame
		capture.read(*frameToBeDisplayed);

		//adding current frame to vector/array list of matricies
		globalFrames.push_back(*frameToBeDisplayed);
		//globalFrames.at(i).convertTo(globalGrayFrames.at(i), CV_8U);
		Mat tmpGrayScale;
		globalFrames.at(i).convertTo(tmpGrayScale, CV_8U);

		cvtColor(globalFrames.at(i), tmpGrayScale, CV_BGR2GRAY);
		globalGrayFrames.push_back(tmpGrayScale);

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

		if(i > 2)
		{
			backgroundFrame = medianImage();
			backgroundFrame = meanImage();
		}

		cout << i << endl;

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
