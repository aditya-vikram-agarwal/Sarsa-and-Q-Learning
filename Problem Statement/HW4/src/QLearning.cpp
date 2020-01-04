#include "stdafx.h"

using namespace std;

// This constructor is called whenever a QLearning object is created. The bit at the end of the line below initializes the private member variables to be the values provided as arguments.
QLearning::QLearning(const int & stateDim, const int & numActions, const double & alpha, const double & gamma, const double & epsilon, const int & iOrder, const int & dOrder) : stateDim(stateDim), numActions(numActions), alpha(alpha), gamma(gamma) {
	// Initialize the FourierBasis, computing the C-matrix.
	fb.init(stateDim, iOrder, dOrder);

	// Get the number of features the FourierBasis will output given the specified stateDim, iOrder, and dOrder
	numFeatures = fb.getNumOutputs();

	// Initialize the weights to be a vector of numActions vectors, each of which has numFeatures elements, all initially zero.
	w.resize(numActions);
	for (int a = 0; a < numActions; a++)
		w[a] = vector<double>(numFeatures, 0.0);

	// Initialize phi (which is phi(s)) and phiPrime (which is phi(sPrime)) to be of length numFeatures, and equal to zero
	phiPrime = phi = vector<double>(numFeatures, 0.0);

	// Set d1 to be a Bernoulli that returns true with probability epsilon.
	d1 = bernoulli_distribution(epsilon);

	// Make d2 a uniform distribution over {0,1,...,numActions-1}.
	d2 = uniform_int_distribution<int>(0, numActions - 1);
}

// Train given an (s,a,r,s') tuple. We won't be using the generator here, since the QLearning update is not random. If sPrimeTerminal==true, then after this call to train, "newEpisode" will be called - we will not train with s set to what is sPrime right now, as all subsequent rewards would be zero.
void QLearning::train(std::mt19937_64 & generator, const std::vector<double> & s, const int & a, double & r, const std::vector<double> & sPrime, const bool & sPrimeTerminal) {
	// If we haven't initialized phi, initialize it and set the flag for phiInit.
	if (!phiInit) {
		phi = fb.basify(s);	// fb.basify(s) returns the features for state s.
		phiInit = true;
	}

	// we know q(terminal_state, any_action) = 0.
	if (!sPrimeTerminal) {
		phiPrime = fb.basify(sPrime);		// Get phi(sPrime)
	}
	
	// TODO: Put your code here for QLearning's update. Hint: you may want to compute the TD error first, then perform the update.
	// Another hint: How you compute the TD-error should likely depend on whether sPrimeTerminal is true!
	// I'm leaving in part of my solution, which makes it clear how I intended for phi and phiPrime to be loaded. For me, this code
	// was within the first step - computing the TD-error. You probably shouldn't be calling fb.basify(sPrime) if sPrimeTerminal == true, since

	double TDerror = r + gamma * maxQ(phiPrime) - dot(w[a], phi);

	if(sPrimeTerminal == true) TDerror = r - dot(w[a], phi);

	for(int i = 0; i < (int)w[a].size(); i++) w[a][i] += alpha * TDerror * phi[i];

	// Your code should probably all be above this comment.
	
	// We computed phi(sPrime). Store that in phi, as this will be phi(s) next time train is called.
	phi = phiPrime;
}

void QLearning::newEpisode(mt19937_64 & generator) {
	// Note that phi has not been initialized during the previous call to train, as this is a new episode and the next
	// call to train will be the first of the episode.
	phiInit = false;
}

int QLearning::getAction(const std::vector<double> & s, std::mt19937_64 & generator) {
	// d1(generator) returns true with probability epsilon.
	if (d1(generator))
		return d2(generator);	// Explore. d2(generator) returns a uniform-random number from 0 to numActions-1 (see the constructor for where this distribution object was initialized)

	// We should act greedily. First, convert s to features (we don't call these "phi", since that is a member variable that we don't want to over-write).
	vector<double> features = fb.basify(s);
	vector<int> bestActions(1, 0);	// Create a vector to store the best actions we have found so far. Put in action a=0.
	double bestActionValue = dot(w[0],features);		// Get q(s,0), and store in bestActionValue.
	for (int a = 1; a < numActions; a++) {				// Loop over actions, starting with a=1, and see if it is better than our currently stored bestActionValue
		double curActionValue = dot(w[a], features);	// Get q(s,a)
		if (curActionValue == bestActionValue)			// if q(s,a) == bestActionValue
			bestActions.push_back(a);						// Append action a to the list of best actions
		else if (curActionValue > bestActionValue) {	// if q(s,a) > bestActionValue
			bestActionValue = curActionValue;				// Set bestActionValue to be q(s,a)
			bestActions.resize(1);							// Empty out bestActions to only have one element
			bestActions[0] = a;								// Set that one element to be action a.
		}
	}
	if ((int)bestActions.size() == 1)					// Is there only one best action?
		return bestActions[0];								// If so, return it. This is the most common case, and avoids using a random number generator most of the time.
	return (uniform_int_distribution<int>(0, (int)bestActions.size() - 1))(generator);	// There are many best actions. Select one uniformly randomly from bestActions.
}

// Return max_{a \in \mathcal A} q(s,a), where phi is phi(s).
double QLearning::maxQ(const vector<double> & phi) const {
	double result = dot(w[0], phi);				// Start with q(s,0)
	for (int a = 1; a < numActions; a++)		// Loop over actions a, starting with action 1
		result = max(result, dot(w[a],phi));	// Set our current max value to be the max of our previous max value and q(s,a).
	return result;								// Return the max value that we found.
}