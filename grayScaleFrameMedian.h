/*
 * grayScaleFrameMedian.h
 *
 *  Created on: Aug 4, 2015
 *      Author: Vidur
 */

#ifndef GRAYSCALEFRAMEMEDIAN_H_
#define GRAYSCALEFRAMEMEDIAN_H_

//thread to calculate median of image
void *calcMedianImage(void *threadarg);

//method to perform median on grayscale images
void grayScaleFrameMedian();

#endif /* GRAYSCALEFRAMEMEDIAN_H_ */
