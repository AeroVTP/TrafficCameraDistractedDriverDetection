/*
 * learnedDistanceFromNormal.cpp
 *
 *  Created on: Aug 5, 2015
 *      Author: Vidur
 */

using namespace std;

void learnedDistanceFromNormal(double distance)
{
	extern double learnedLASMDistance;
	extern double learnedLASMDistanceSum;
	extern double learnedLASMDistanceAccess;

	if(distance != 0)
	{
		cout << "distance " << distance << endl;
		cout << "learnedDistance " << learnedLASMDistance << endl;
	}

	distance = abs(distance);

	learnedLASMDistanceAccess++;
	learnedLASMDistanceSum += distance;
	learnedLASMDistance = learnedLASMDistanceSum / learnedLASMDistance;

	if(distance > learnedLASMDistance * 1.25 || distance < learnedLASMDistance * .75)
	{
		extern int detectStrength;
		detectStrength++;
	}
}


