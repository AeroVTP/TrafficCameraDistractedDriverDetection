/*
 * mogDetection2.h
 *
 *  Created on: Aug 4, 2015
 *      Author: Vidur
 */

#ifndef MOGDETECTION2_H_
#define MOGDETECTION2_H_

void mogDetection2ThreadHandler(bool buffer);
void *computeBgMog2(void *threadarg);

#endif /* MOGDETECTION2_H_ */
