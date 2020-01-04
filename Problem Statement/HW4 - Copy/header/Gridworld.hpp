#pragma once

#include <stdafx.h>

// A "Gridworld" object is an environment that the agent can interact with.
class Gridworld {
public:	// This means that code outside of this class can see and reference the following functions.
	// This is the "constructor". This function is called whenever a Gridworld object is created.
	Gridworld();

	// Below is a function. If X is a Gridworld object, you would call this function with X.getStateDim(). The "const" after
	// the function says that this function will not change any of the member variables (variables associated with this object, defined below).
	// This particular function returns the dimension of the state (which is passed as a vector).
	int getStateDim() const;

	// The the number of discrete actions.
	int getNumActions() const;

	// Update the state of the environment based on the provided action. We are given a random number
	// generator to use in case we need to sample any random numbers to compute the state transition. Notice
	// that this function is not "const", since it will change the state.
	double update(const int & action, std::mt19937_64 & generator);

	// Get the curretn state, as a vector object. None of these MDPs we use have noise in the observation,
	// so we won't be using the generator here, but it's passed in case you want to add noise to the state
	// observations.
	// ****** IMPORTANT: The environment will return a NORMALIZED state - a vector that already has
	// all elements in the interval [0,1] (roughly) **********
	std::vector<double> getState(std::mt19937_64 & generator);

	// A function that returns true if the current state is terminal.
	bool inTerminalState() const;

	// Tell the environment to start a new episode. The random number generator is provided so that you
	// can sample from d_0, the initial state distribution, if the initial state is not deterministic.
	void newEpisode(std::mt19937_64 & generator);

private:	// This means that the objects below are not visible to code outside of this class.
	const int size = 5;	// This is the size of the gridworld - it is a 5x5 grid.
	int x, y;			// Agent position. This will be converted into the state.
};