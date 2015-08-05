/*
 * learnedDistanceFromNormal.cpp
 *
 *  Created on: Aug 5, 2015
 *      Author: Vidur
 */
#include "welcome.h"

using namespace std;

void learnedDistanceFromNormal(double distance)
{
	extern double learnedLASMDistance;
	extern double learnedLASMDistanceSum;
	extern double learnedLASMDistanceAccess;

	distance = abs(distance);

	learnedLASMDistanceAccess++;
	learnedLASMDistanceSum += distance;
	learnedLASMDistance = learnedLASMDistanceSum / learnedLASMDistanceAccess;

	const double scalarFactor = distance / learnedLASMDistance;

	//cout << "Distance " << distance << endl;
	//cout << "Scalar Factor " << scalarFactor << endl;

	if(scalarFactor  >  1.25)
	{
		extern int detectStrength;
		extern int i;
		extern int numberOfAnomaliesDetected;

		detectStrength++;
		cout << "LASM FIRING" << endl;
		numberOfAnomaliesDetected++;
		welcome("LASM & LASME ANOMALY -> CONFIRMING SHORTLY FN: " + to_string(i));
	}
}


