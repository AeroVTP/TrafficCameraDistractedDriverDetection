/*
 * gaussianMixtureModel.h
 *
 *  Created on: Aug 4, 2015
 *      Author: Vidur
 */

#ifndef GAUSSIANMIXTUREMODEL_H_
#define GAUSSIANMIXTUREMODEL_H_

using namespace cv;

//method to handle GMM thread
Mat gaussianMixtureModel();
void *calcGaussianMixtureModel(void *threadarg);

#endif /* GAUSSIANMIXTUREMODEL_H_ */
