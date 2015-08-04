/*
 * opticalFlowFarneback.h
 *
 *  Created on: Aug 4, 2015
 *      Author: Vidur
 */

#ifndef OPTICALFLOWFARNEBACK_H_
#define OPTICALFLOWFARNEBACK_H_

//method to handle OFA thread
Mat opticalFlowFarneback();
void *computeOpticalFlowAnalysisObjectDetection(void *threadarg);

#endif /* OPTICALFLOWFARNEBACK_H_ */
