#pragma once

#include <stdafx.h>

class Sarsa {
public:
	Sarsa(const int & stateDim, const int & numActions, const double & alpha, const double & gamma, const double & epsilon, const int & iOrder, const int & dOrder);
	void train(std::mt19937_64 & generator, const std::vector<double> & s, const int & a, double & r, const std::vector<double> & sPrime, const bool & sPrimeTerminal);
	void newEpisode(std::mt19937_64 & generator);
	int getAction(const std::vector<double> & s, std::mt19937_64 & generator);

private:
	FourierBasis fb;
	std::vector<std::vector<double>> w;
	int stateDim, numFeatures, numActions;
	double alpha, gamma;
	std::bernoulli_distribution d1;
	std::uniform_int_distribution<int> d2;
	
	// HERE: You may want to add additional member variables, perhaps storing previous states, features, actions, and/or rewards,
	// along with Boolean flags indicating if they have been initialized.

	std::vector<double> phi_s;
	bool flag = false;
	int previous_a;
	double previous_r;

};
