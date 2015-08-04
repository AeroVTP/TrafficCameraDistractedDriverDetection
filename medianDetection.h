/*
 * medianDetection.h
 *
 *  Created on: Aug 4, 2015
 *      Author: Vidur
 */

#ifndef MEDIANDETECTION_H_
#define MEDIANDETECTION_H_

void medianDetectionThreadHandler(int FRAME_RATE);
void *computeMedianDetection(void *threadarg);
//method to handle median image subtraction
Mat medianImageSubtraction(int FRAME_RATE);

#endif /* MEDIANDETECTION_H_ */
