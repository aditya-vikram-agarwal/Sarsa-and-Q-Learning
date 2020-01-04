#include "stdafx.h"

using namespace std;

/*
This file contains the implementations of all of the functions defined in the Gridworld.hpp file.
*/

// Implement the constructor.
Gridworld::Gridworld() {
	mt19937_64 generator(0);	// Initialize the RNG.
	newEpisode(generator);		// Start a new episode.
}

int Gridworld::getStateDim() const {
	return size*size;			// The state will be a tabular representation, implemented using linear function approxiamtion.
}

int Gridworld::getNumActions() const {
	return 4;					// up/down/left/right
}

double Gridworld::update(const int & action, mt19937_64 & generator) {
	// Actions correspond to up/down/left/right, where (0,0) is bottom left. Actions always succeed
	if (action == 0)
		y++;
	else if (action == 1)
		y--;
	else if (action == 2)
		x--;
	else
		x++;
	x = bound(x, 0, size - 1);	// The bound function is defined in MathUtils.hpp
	y = bound(y, 0, size - 1);	
	return -1;	// Reward is always -1
}

vector<double> Gridworld::getState(mt19937_64 & generator) {
	vector<double> result(size*size, 0.0);	// Effective tabular representation, one element per state, all set to zero.
	result[x + y*size] = 1.0;				// Set the s'th element to be 1, where we map x-y coordinates to unique integers.	
	return result;
}

bool Gridworld::inTerminalState() const {
	return ((x == size - 1) && (y == size - 1));	// Are we in state (size-1,size-1)?
}

void Gridworld::newEpisode(mt19937_64 & generator) {
	x = y = 0;								// Always start in state (0,0).
}