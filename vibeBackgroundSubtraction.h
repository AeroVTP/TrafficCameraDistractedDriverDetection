/*
 * vibeBackgroundSubtraction.h
 *
 *  Created on: Aug 4, 2015
 *      Author: Vidur
 */

#ifndef VIBEBACKGROUNDSUBTRACTION_H_
#define VIBEBACKGROUNDSUBTRACTION_H_

void vibeBackgroundSubtractionThreadHandler(bool buffer);
void *computeVibeBackgroundThread(void *threadarg);

#endif /* VIBEBACKGROUNDSUBTRACTION_H_ */
