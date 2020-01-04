#include "stdafx.h"

using namespace std;

MountainCar::MountainCar() {
	state.resize(2);
	mt19937_64 generator(0);
	newEpisode(generator);
}

int MountainCar::getStateDim() const {
	return 2;
}

int MountainCar::getNumActions() const {
	return 3;
}

double MountainCar::update(const int & action, mt19937_64 & generator) {
	double u = (double)action - 1.0;	// Convert act to a double in {-1, 0, 1}
	// Update xDot and then x
	state[1] = bound(state[1] + 0.001*u - 0.0025*cos(3.0*state[0]), minXDot, maxXDot);
	state[0] += state[1];
	if (state[0] < minX) {
		state[0] = minX;
		state[1] = 0;					// Inelastic collisions
	}
	if (state[0] > maxX)
		state[0] = maxX;
	return -1;							// Reward is always -1
}

vector<double> MountainCar::getState(mt19937_64 & generator) {
	vector<double> result(2);
	result[0] = normalize(state[0], minX, maxX);
	result[1] = normalize(state[1], minXDot, maxXDot);
	return result;
}

bool MountainCar::inTerminalState() const {
	return state[0] >= maxX;
}

void MountainCar::newEpisode(mt19937_64 & generator) {
	state[0] = -0.5;
	state[1] = 0;
}