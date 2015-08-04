/*
 * blurFrame.h
 *
 *  Created on: Aug 4, 2015
 *      Author: Vidur
 */

#ifndef BLURFRAME_H_
#define BLURFRAME_H_

//method to blur Mat using custom kernel size
Mat blurFrame(string blurType, Mat sourceDiffFrame, int blurSize);

#endif /* BLURFRAME_H_ */
