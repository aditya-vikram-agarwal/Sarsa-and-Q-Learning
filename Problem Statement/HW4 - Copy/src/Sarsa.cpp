#include "stdafx.h"

using namespace std;

// This is the constructor. If you added member variables, be sure to initialize them here.
Sarsa::Sarsa(const int & stateDim, const int & numActions, const double & alpha, const double & gamma, const double & epsilon, const int & iOrder, const int & dOrder) : stateDim(stateDim), numActions(numActions), alpha(alpha), gamma(gamma) {
	fb.init(stateDim, iOrder, dOrder);
	numFeatures = fb.getNumOutputs();
	w.resize(numActions);
	for (int a = 0; a < numActions; a++)
		w[a] = vector<double>(numFeatures, 0.0);
	d1 = bernoulli_distribution(epsilon);
	d2 = uniform_int_distribution<int>(0, numActions - 1);
}

// This is the train function. While the contents will differ from QLearning, you might copy the general structure (if-statements checking that terms are initialized, compute TD-error, update weights, set cur <-- new (curState, curAction, curReward?)
void Sarsa::train(std::mt19937_64 & generator, const std::vector<double> & s, const int & a, double & r, const std::vector<double> & sPrime, const bool & sPrimeTerminal) {
	
	std::vector<double> s_dash = s;
	std::vector<double> phi_s_dash = fb.basify(s_dash);

	if (flag == true) {

		double term3 = dot(w[previous_a], phi_s);
		double term2 = gamma * dot(w[a], phi_s_dash);
		double TDerror = previous_r + term2 - term3;

		for (int i = 0; i < (int)w[previous_a].size(); i++)	w[previous_a][i] += alpha * TDerror * phi_s[i];

		if (sPrimeTerminal == true) {
			double term3 = dot(w[a], phi_s_dash);
			double term2 = gamma * 0.0;
			double TDerror = r + term2 - term3;
			for (int i = 0; i < (int)w[a].size(); i++) w[a][i] += alpha * TDerror * phi_s_dash[i];
		}
	}

	flag = true;
	previous_a = a;
	previous_r = r;
	phi_s = phi_s_dash;

}

// When a new episode starts, do you need to clear any of your variables, or set any of your flags to true/false?
void Sarsa::newEpisode(mt19937_64 & generator) {
	flag = false;
}

// This is identicaly to the getAction function in QLearning. You shouldn't have to change this.
int Sarsa::getAction(const std::vector<double> & s, std::mt19937_64 & generator) {
	if (d1(generator)) // Explore
		return d2(generator);
	vector<double> features = fb.basify(s);
	vector<int> bestActions(1, 0);
	double bestActionValue = dot(w[0],features);
	for (int a = 1; a < numActions; a++) {
		double curActionValue = dot(w[a], features);
		if (curActionValue == bestActionValue)
			bestActions.push_back(a);
		else if (curActionValue > bestActionValue) {
			bestActionValue = curActionValue;
			bestActions.resize(1);
			bestActions[0] = a;
		}
	}
	if ((int)bestActions.size() == 1)
		return bestActions[0];
	return (uniform_int_distribution<int>(0, (int)bestActions.size() - 1))(generator);
}