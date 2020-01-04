#pragma once

#include "stdafx.h"

/*
This is a QLearning agent. You may change this file, but you do not need to (this is the file in our solution, without changes).
You may add member variables under "private", but should not change the public functions.
*/
class QLearning {
public:
	// This is the constructor. It initializes this agent for an MDP with states provided as vectors of length stateDim, numActions discrete actions,
	// a step size of alpha, discount parameter gamma, epsilon-greedy exploration parameter epsilon, and using the FourierBasis with independent (decoupled) order iOrder,
	// and dependent (coupled) order dOrder.
	QLearning(const int & stateDim, const int & numActions, const double & alpha, const double & gamma, const double & epsilon, const int & iOrder, const int & dOrder);

	// Train the agent based on the transition s,a,r,sPrime (with sPrimeTerminal indicating if sPrime is a terminal state, meaning we will never run the update with s = sPrime).
	void train(std::mt19937_64 & generator, const std::vector<double> & s, const int & a, double & r, const std::vector<double> & sPrime, const bool & sPrimeTerminal);

	// Tell the agent we are starting a new episode.
	void newEpisode(std::mt19937_64 & generator);

	// As the agent to provide an action given that we are in state s.
	int getAction(const std::vector<double> & s, std::mt19937_64 & generator);

private:
	// This object, once initialized, takes in state-vectors and outputs feature vectors constructed using the Fourier Basis.
	FourierBasis fb;

	// The weight vector for linear q-approximation. We store it as one vector for each action. So, w[numActions][numFeatures]. q(s,a) = dot product of w[a] with phi(s).
	std::vector<std::vector<double>> w;

	// Properties of the MDP
	int stateDim, numFeatures, numActions;

	// Step size, and the gamma-hyperparameter for the Q-Learning algorithm (may or may not match the one used when reporting results)
	double alpha, gamma;

	// phi and phiPrime are phi(s) and phi(s'). We store them so that, between calls to train, we don't recompute phi(s) when we computed it as phi(sPrime) at the previous time step.
	bool phiInit = false;				// Has phi been initialized? False at t=0, true thereafter.
	std::vector<double> phi, phiPrime;

	// A Bernoulli distribution for determining if we should act greedily or uniformly randomly (we use epsilon greedy)
	std::bernoulli_distribution d1;

	// A uniform distribution over actions for when we choose to explore.
	std::uniform_int_distribution<int> d2;

	// Compute max_{a} q(s,a). This function takes the features phi(s) rather than s directly.
	double maxQ(const std::vector<double> & phi) const;
};
