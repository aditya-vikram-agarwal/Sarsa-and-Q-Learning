#pragma once

#include "stdafx.h"

// MountainCar MDP - see Gridworld.hpp for comments regarding the general structure of these environment/MDP objects
class MountainCar {
public:
	MountainCar();	
	int getStateDim() const;
	int getNumActions() const;
	double update(const int & action, std::mt19937_64 & generator);
	std::vector<double> getState(std::mt19937_64 & generator);
	bool inTerminalState() const;
	void newEpisode(std::mt19937_64 & generator);

private:
	const double minX = -1.2;
	const double maxX = 0.5;
	const double minXDot = -0.07;
	const double maxXDot = 0.07;

	std::vector<double> state;	// [2] - x and xDot
};