#pragma once

#include <stdafx.h>

// Acrobot MDP - see Gridworld.hpp for comments regarding the general structure of these environment/MDP objects
class Acrobot {
public:
	Acrobot();
	int getStateDim() const;
	int getNumActions() const;
	double update(const int & action, std::mt19937_64 & generator);
	std::vector<double> getState(std::mt19937_64 & generator);
	bool inTerminalState() const;
	void newEpisode(std::mt19937_64 & generator);

private:
	// Standard parameters for the acrobot domain
	const double m1 = 1;				// Mass of the first link
	const double m2 = 1;				// Mass of the second link
	const double l1 = 1;				// Length of the first link
	const double l2 = 1;				// Length of the second link
	const double lc1 = .5;
	const double lc2 = .5;
	const double i1 = 1;
	const double i2 = 1;
	const double g = 9.8;				// Acceleration due to gravity
	const double fmax = 1;				// Maximum (and -minimum) force that can be applied
	const double dt = 0.2;				// Time step duration
	const double integShritte = 10;		// The larger this is, the more accurate the Runge-Kutta approximation

	double t;							// Time into the episode
	double theta1;						// Joint angle closest to the pivot
	double theta2;						// Joint angle of the 'elbow'
	double theta1Dot;					// Time derivative of theta1
	double theta2Dot;					// Time derivative of theta2

	// Function used by the Runge-Kutta approximation
	void f(double s[4], double tau, double * buff);

	// Vars used by Runge-Kutta approx
	double s0_dot[4];
	double s1_dot[4];
	double s2_dot[4];
	double s3_dot[4];
	double hilf[4];
	double ss[4];
	double s1[4];
	double s2[4];
	double s3[4];
};