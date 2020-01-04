#pragma once

#include <stdafx.h>

// Cart-Pole MDP - see Gridworld.hpp for comments regarding the general structure of these environment/MDP objects
class CartPole {
public:
	CartPole();
	int getStateDim() const;
	int getNumActions() const;
	double update(const int & action, std::mt19937_64 & generator);
	std::vector<double> getState(std::mt19937_64 & generator);
	bool inTerminalState() const;
	void newEpisode(std::mt19937_64 & generator);

private:
	// Standard parameters for the CartPole domain
	const int simSteps = 10;
	const double dt = 0.02;
	const double uMax = 10.0;
	const double l = 0.5;
	const double g = 9.8;
	const double m = 0.1;
	const double mc = 1;
	const double muc = 0.0005;
	const double mup = 0.000002;
	
	// State variables ranges
	const double xMin = -2.4;
	const double xMax = 2.4;
	const double vMin = -10;
	const double vMax = 10;
	const double thetaMin = -M_PI / 12.0;
	const double thetaMax = M_PI / 12.0;
	const double omegaMin = -M_PI;
	const double omegaMax = M_PI;

	// State variables
	double x;
	double v;
	double theta;
	double omega;
	double t;
};