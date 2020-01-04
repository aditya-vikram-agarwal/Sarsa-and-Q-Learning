// We have put all of our #include statements into this one file. By including this one file in our
// other files, we are therefore including everything.
#include <stdafx.h>

// This let's us not have to write std::vector all the time.
using namespace std;

// This is a "templated" function. Here "Agent" and "Environment" can be any objects that allow this function to compile.
// The compler will work out all objects "Agent" and "Environment" that this function is called with, and will compile
// different versions for each. This allows us to pass different objects as the "Environment". See in runMountainCar
// and runCartPole how we call this function with different objects for the first argument (Agent could be Sarsa or QLearning
// objects) and different second arguments (all four MDPs that we coded up).
// This functionality could also be achieved with one Environment class with different environments as sub-classes. 
//
// This function runs "numTrials" agent lifetimes, each containing numEpisodes episodes, on the provided environment. 
// Episodes are terminated after maxEpisodeLength timesteps. The gamma here is the one used when plotting expected
// returns (this isn't the gamme provided to the agent as a hyper-parameter). The object mt19937_64 is a random number
// generator (see std::random).
//
// This function outputs two things, so it's easier to make them arguments that are passed "by reference" (with the &), meaning
// that if this function changes their value, the calling function will see these changed values. These two "outputs" are
// meanBuff and varBuff. These arrays record the mean discounted return for each episode number, and the sample variance of
// the discounted returns for each episode number. That is, meanBuff's length is numEpisodes, and meanBuff[i] is the average
// return on the i'th episode across the numTrials trials. varBuff[i] is the variance of the returns during the i'th episodes
// from the numTrials trials.
template <typename Agent, typename Environment>
void runExperiment(Agent & a, Environment & e, const int & numTrials, const int & numEpisodes, const int & maxEpisodeLength, const double & gamma, mt19937_64 & generator, vector<double> & meanBuff, vector<double> & varBuff) {
	/*
	This function is multithreaded. To avoid having two threads over-writing the same result locations in memory, we will create separate objects and places to store
	results for every thread. We will have roughly one thread per trial (capped at your number of hyperthreads for your CPU).
	*/
	vector<vector<double>> returns(numTrials);		// We will run numTrials, each lasting numEpisodes. Store the returns from every episode in this object, which is a vector of vectors (really a matrix, as all will be the same length).
	vector<Agent> agents(numTrials, a);				// Create numTrials copies of the agent. This "constructor" for a std::vector object sets every element equal to the second argument, in this case, 'a', the agent passed in.
	vector<Environment> environments(numTrials, e);	// Similarly, make numTrials copies of the environment, one for each thread.
	vector<mt19937_64> generators(numTrials);		// Create numTrials random number generators. Don't make them all equal though! The loop below seeds them all differently.
	for (int trial = 0; trial < numTrials; trial++)
		generators[trial].seed(trial);
	#pragma omp parallel for						// Ignore this line. It is the magic that makes the following for-loop happen in parallel.
	for (int trial = 0; trial < numTrials; trial++) {	// Loop over trials
		// std::printf("%d", trial);
		returns[trial] = vector<double>(numEpisodes, 0.0);	// Resize the trial'th returns array to be of length numEpisodes, and set all entries equal to zero. (Recall the first line made returns a vector of length numTrials, essentially setting the number of rows - here we are setting the number of columns).
		vector<double> state, nextState; // The current state and the next state, as vectors. Put outside loop to only allocate once
		for (int episode = 0; episode < numEpisodes; episode++) {	// Loop over episodes
			double curGamma = 1.0;					// We plot the discounted return - this stores gamma^t, which starts at 1.
			bool inTerminalState = false;			// We will use this flag to determine when we should terminate the loop below. If environment[trial].inTerminalState() is slow to call, this saves us from calling it a couple times. For our MDPs it really doesn't matter that we're doing this more efficiently.
			environments[trial].newEpisode(generators[trial]);	// Reset the environment, telling it to start a new episode.
			agents[trial].newEpisode(generators[trial]);		// Tell the agent that we are starting a new episode. 
			state = environments[trial].getState(generators[trial]);	// Get teh initial state.
			for (int t = 0; (t < maxEpisodeLength) && (!inTerminalState); t++) {	// Loop over time steps in the episode, stopping when we hit the max episode length or when we enter a terminal state.
				int action = agents[trial].getAction(state, generators[trial]);		// Get the current action
				double reward = environments[trial].update(action, generators[trial]);	// Apply the action by updating the environment with the chosen action, and get the resulting reward.
				returns[trial][episode] += curGamma * reward;							// Update the expected return for the current episode.
				nextState = environments[trial].getState(generators[trial]);			// Get the resulting state of the environment from this transition
				inTerminalState = environments[trial].inTerminalState();				// Store whether this is next-state is a terminal state.
				agents[trial].train(generators[trial], state, action, reward, nextState, inTerminalState);	// Update the agent, telling it if "nextState" is a terminal state.
				state = nextState;														// Prepare for the next iteration of the loop with this line and the next.
				curGamma *= gamma;
			}
		}
	}
	// Clear the two buffers that we will use for output, setting them both to be of length numEpisodes, and initialized to zero
	meanBuff = varBuff = vector<double>(numEpisodes, 0.0);
	vector<double> cur(numTrials);	// This array will store all of the returns from the epCount'th episode across all trials
	for (int epCount = 0; epCount < numEpisodes; epCount++) {	// Loop over episodes
		for (int trial = 0; trial < numTrials; trial++)			// Loop over trials
			cur[trial] = returns[trial][epCount];				// Store in cur[trial] all of the returns from the epCount'th episode.
		meanBuff[epCount] = mean(cur);							// Get the mean of cur.
		varBuff[epCount] = var(cur);							// Get the variance of cur.
	}
}

// Run Q-learning and Sarsa on Mountain Car.
void runMountainCar() {
	mt19937_64 generator(0);	// Create the random number generator.
	int numTrials = 100, numEpisodes = 40, maxEpisodeLength = 20000;	// numTrials = number of agent lifetimes, numEpisodes is the number of episodes in an agent lifetime, and episodes are always terminated if they reach time step maxEpisodeLength.
	MountainCar e;				// Create the environment object, in this case a MountainCar object.
	double gamma = 1.0;			// Plot expected returns with this value of gamma (you might use the same parameter in your agent, or you might not!)
	
	// Create the two agents. The arguments here are the hyperparameters that you must tune! The ones we provide you below
	// are bad first examples.
	//													alpha		gamma	epsilon	iOrder	dOrder
	QLearning a1(e.getStateDim(), e.getNumActions(),	0.00001,	0,		1,		1,		1);
	Sarsa a2(e.getStateDim(), e.getNumActions(),		70,			1,		0.95,	5,		0);
	
	// Run each agent on the mountain car environment using the runEnvironment function (see above for a description of what it stores in the last
	// two arguments (means and vars).
	vector<double> means1, vars1, means2, vars2;
	runExperiment(a1, e, numTrials, numEpisodes, maxEpisodeLength, gamma, generator, means1, vars1);
	runExperiment(a2, e, numTrials, numEpisodes, maxEpisodeLength, gamma, generator, means2, vars2);

	// Dump the results of the experiment to an output csv file. The first column will be the episode number, the second will be the mean
	// discounted return for Q-learning, the third column will be the mean discounted return for Sarsa, the fourth column will be the standard
	// deviation of the discounted returns for Q-learning (which you will use for error bars), and the fifth (final) column will be the standard
	// deviation of the discounted returns for Sarsa (also used for error bars).
	ofstream out("../../../output/out_MountainCar.csv");
	out << "Number of Episodes,"
		<< "Q-Learning,Sarsa,"
		<< "Stddev Q-Learning,Stddev Sarsa" << endl;
	for (int epCount = 0; epCount < numEpisodes; epCount++) {
		out << epCount << ","
			<< means1[epCount] << "," << means2[epCount] << "," 
			<< sqrt(vars1[epCount]) << "," << sqrt(vars2[epCount]) << endl;
	}
	out.close();
}

// Run Q-learning and Sarsa on Mountain Car.
void runMountainCarwParamQ(double a, double g, double ee, int i, int d) {
	mt19937_64 generator(0);	// Create the random number generator.
	int numTrials = 100, numEpisodes = 40, maxEpisodeLength = 20000;	// numTrials = number of agent lifetimes, numEpisodes is the number of episodes in an agent lifetime, and episodes are always terminated if they reach time step maxEpisodeLength.
	MountainCar e;				// Create the environment object, in this case a MountainCar object.
	double gamma = 1.0;			// Plot expected returns with this value of gamma (you might use the same parameter in your agent, or you might not!)
	
	// Create the two agents. The arguments here are the hyperparameters that you must tune! The ones we provide you below
	// are bad first examples.
	//													alpha		gamma	epsilon	iOrder	dOrder
	QLearning a1(e.getStateDim(), e.getNumActions(),	a,g,ee,i,d);
	Sarsa a2(e.getStateDim(), e.getNumActions(),		a,g,ee,i,d);
	
	// Run each agent on the mountain car environment using the runEnvironment function (see above for a description of what it stores in the last
	// two arguments (means and vars).
	vector<double> means1, vars1;
	vector<double> means2(numEpisodes,0.0);
	vector<double> vars2(numEpisodes,0.0);
	runExperiment(a1, e, numTrials, numEpisodes, maxEpisodeLength, gamma, generator, means1, vars1);
	// runExperiment(a2, e, numTrials, numEpisodes, maxEpisodeLength, gamma, generator, means2, vars2);

	// Dump the results of the experiment to an output csv file. The first column will be the episode number, the second will be the mean
	// discounted return for Q-learning, the third column will be the mean discounted return for Sarsa, the fourth column will be the standard
	// deviation of the discounted returns for Q-learning (which you will use for error bars), and the fifth (final) column will be the standard
	// deviation of the discounted returns for Sarsa (also used for error bars).
	ofstream out("../../../output/"+to_string(numEpisodes)+"out_Mountain-"+to_string(a)+"-"+to_string(g)+"-"+to_string(ee)+"-"+to_string(i)+"-"+to_string(d)+"-"+to_string(means1[numEpisodes-1])+"qlearning.csv");
	out << "Number of Episodes,"
		<< "Q-Learning,Sarsa,"
		<< "Stddev Q-Learning,Stddev Sarsa" << endl;
	for (int epCount = 0; epCount < numEpisodes; epCount++) {
		out << epCount << ","
			<< means1[epCount] << "," << means2[epCount] << "," 
			<< sqrt(vars1[epCount]) << "," << sqrt(vars2[epCount]) << endl;
	}
	out.close();
	cout << to_string(means1[numEpisodes-1]);
}

// Run Q-learning and Sarsa on Mountain Car.
void runMountainCarwParamS(double a, double g, double ee, int i, int d) {
	mt19937_64 generator(0);	// Create the random number generator.
	int numTrials = 100, numEpisodes = 40, maxEpisodeLength = 20000;	// numTrials = number of agent lifetimes, numEpisodes is the number of episodes in an agent lifetime, and episodes are always terminated if they reach time step maxEpisodeLength.
	MountainCar e;				// Create the environment object, in this case a MountainCar object.
	double gamma = 1.0;			// Plot expected returns with this value of gamma (you might use the same parameter in your agent, or you might not!)
	
	// Create the two agents. The arguments here are the hyperparameters that you must tune! The ones we provide you below
	// are bad first examples.
	//													alpha		gamma	epsilon	iOrder	dOrder
	QLearning a1(e.getStateDim(), e.getNumActions(),	a,g,ee,i,d);
	Sarsa a2(e.getStateDim(), e.getNumActions(),		a,g,ee,i,d);
	
	// Run each agent on the mountain car environment using the runEnvironment function (see above for a description of what it stores in the last
	// two arguments (means and vars).
	vector<double> means2, vars2;
	vector<double> means1(numEpisodes,0.0);
	vector<double> vars1(numEpisodes,0.0);
	// runExperiment(a1, e, numTrials, numEpisodes, maxEpisodeLength, gamma, generator, means1, vars1);
	runExperiment(a2, e, numTrials, numEpisodes, maxEpisodeLength, gamma, generator, means2, vars2);

	// Dump the results of the experiment to an output csv file. The first column will be the episode number, the second will be the mean
	// discounted return for Q-learning, the third column will be the mean discounted return for Sarsa, the fourth column will be the standard
	// deviation of the discounted returns for Q-learning (which you will use for error bars), and the fifth (final) column will be the standard
	// deviation of the discounted returns for Sarsa (also used for error bars).
	ofstream out("../../../output/"+to_string(numEpisodes)+"out_Mountain-"+to_string(a)+"-"+to_string(g)+"-"+to_string(ee)+"-"+to_string(i)+"-"+to_string(d)+"-"+to_string(means2[numEpisodes-1])+"sarsa.csv");
	out << "Number of Episodes,"
		<< "Q-Learning,Sarsa,"
		<< "Stddev Q-Learning,Stddev Sarsa" << endl;
	for (int epCount = 0; epCount < numEpisodes; epCount++) {
		out << epCount << ","
			<< means1[epCount] << "," << means2[epCount] << "," 
			<< sqrt(vars1[epCount]) << "," << sqrt(vars2[epCount]) << endl;
	}
	out.close();
	cout << to_string(means2[numEpisodes-1]);
}

// See runMountainCar: This is the same thing, but for the CartPole environment.
void runCartPole() {
	mt19937_64 generator(0);
	int numTrials = 50, numEpisodes = 50, maxEpisodeLength = INT_MAX;
	CartPole e;
	double gamma = 1.0;
	//													alpha		gamma	epsilon	iOrder	dOrder
	QLearning a1(e.getStateDim(), e.getNumActions(),	0.00001,	0,		1,		1,		1);
	Sarsa a2(e.getStateDim(), e.getNumActions(),		70,			1,		0.95,	4,		0);
	vector<double> means1, vars1, means2, vars2;
	runExperiment(a1, e, numTrials, numEpisodes, maxEpisodeLength, gamma, generator, means1, vars1);
	runExperiment(a2, e, numTrials, numEpisodes, maxEpisodeLength, gamma, generator, means2, vars2);
	ofstream out("../../../output/out_CartPole.csv");
	out << "Number of Episodes,"
		<< "Q-Learning,Sarsa,"
		<< "Stddev Q-Learning,Stddev Sarsa" << endl;
	for (int epCount = 0; epCount < numEpisodes; epCount++) {
		out << epCount << ","
			<< means1[epCount] << "," << means2[epCount] << "," 
			<< sqrt(vars1[epCount]) << "," << sqrt(vars2[epCount]) << endl;
	}
	out.close();
}

// See runMountainCar: This is the same thing, but for the CartPole environment.
void runCartPolewParamQ(double a, double g, double ee, int i, int d) {
	mt19937_64 generator(0);
	int numTrials = 50, numEpisodes = 50, maxEpisodeLength = INT_MAX;
	CartPole e;
	double gamma = 1.0;
	//													alpha		gamma	epsilon	iOrder	dOrder
	QLearning a1(e.getStateDim(), e.getNumActions(),	a,g,ee,i,d);
	Sarsa a2(e.getStateDim(), e.getNumActions(),		a,g,ee,i,d);
	vector<double> means1, vars1;
	vector<double> means2(numEpisodes,0.0);
	vector<double> vars2(numEpisodes,0.0);
	runExperiment(a1, e, numTrials, numEpisodes, maxEpisodeLength, gamma, generator, means1, vars1);
	// runExperiment(a2, e, numTrials, numEpisodes, maxEpisodeLength, gamma, generator, means2, vars2);
	ofstream out("../../../output/"+to_string(numEpisodes)+"out_CartPole-"+to_string(a)+"-"+to_string(g)+"-"+to_string(ee)+"-"+to_string(i)+"-"+to_string(d)+"-"+to_string(means1[numEpisodes-1])+"qlearning.csv");
	out << "Number of Episodes,"
		<< "Q-Learning,Sarsa,"
		<< "Stddev Q-Learning,Stddev Sarsa" << endl;
	for (int epCount = 0; epCount < numEpisodes; epCount++) {
		out << epCount << ","
			<< means1[epCount] << "," << means2[epCount] << "," 
			<< sqrt(vars1[epCount]) << "," << sqrt(vars2[epCount]) << endl;
	}
	out.close();
	cout << to_string(means1[numEpisodes-1]);
}

// See runMountainCar: This is the same thing, but for the CartPole environment.
void runCartPolewParamS(double a, double g, double ee, int i, int d) {
	mt19937_64 generator(0);
	int numTrials = 50, numEpisodes = 50, maxEpisodeLength = INT_MAX;
	CartPole e;
	double gamma = 1.0;
	//													alpha		gamma	epsilon	iOrder	dOrder
	QLearning a1(e.getStateDim(), e.getNumActions(),	a,g,ee,i,d);
	Sarsa a2(e.getStateDim(), e.getNumActions(),		a,g,ee,i,d);
	vector<double> means2, vars2;
	vector<double> means1(numEpisodes,0.0);
	vector<double> vars1(numEpisodes,0.0);
	// runExperiment(a1, e, numTrials, numEpisodes, maxEpisodeLength, gamma, generator, means1, vars1);
	runExperiment(a2, e, numTrials, numEpisodes, maxEpisodeLength, gamma, generator, means2, vars2);
	ofstream out("../../../output/"+to_string(numEpisodes)+"out_CartPole-"+to_string(a)+"-"+to_string(g)+"-"+to_string(ee)+"-"+to_string(i)+"-"+to_string(d)+"-"+to_string(means2[numEpisodes-1])+"sarsa.csv");
	out << "Number of Episodes,"
		<< "Q-Learning,Sarsa,"
		<< "Stddev Q-Learning,Stddev Sarsa" << endl;
	for (int epCount = 0; epCount < numEpisodes; epCount++) {
		out << epCount << ","
			<< means1[epCount] << "," << means2[epCount] << "," 
			<< sqrt(vars1[epCount]) << "," << sqrt(vars2[epCount]) << endl;
	}
	out.close();
	cout << to_string(means2[numEpisodes-1]);
}

// See runMountainCar: This is the same thing, but for the Acrobot environment.
void runAcrobot() {
	mt19937_64 generator(0);
	int numTrials = 100, numEpisodes = 100, maxEpisodeLength = 3000;
	double gamma = 1.0;
	Acrobot e;
	//													alpha		gamma	epsilon	iOrder	dOrder
	QLearning a1(e.getStateDim(), e.getNumActions(),	0.00001,	0,		1,		1,		1);
	Sarsa a2(e.getStateDim(), e.getNumActions(),		70,			1,		0.95,	2,		0);
	vector<double> means1, vars1, means2, vars2;
	runExperiment(a1, e, numTrials, numEpisodes, maxEpisodeLength, gamma, generator, means1, vars1);
	runExperiment(a2, e, numTrials, numEpisodes, maxEpisodeLength, gamma, generator, means2, vars2);
	ofstream out("../../../output/out_Acrobot.csv");
	out << "Number of Episodes,"
		<< "Q-Learning,Sarsa,"
		<< "Stddev Q-Learning,Stddev Sarsa" << endl;
	for (int epCount = 0; epCount < numEpisodes; epCount++) {
		out << epCount << ","
			<< means1[epCount] << "," << means2[epCount] << "," 
			<< sqrt(vars1[epCount]) << "," << sqrt(vars2[epCount]) << endl;
	}
	out.close();
}

// See runMountainCar: This is the same thing, but for the Acrobot environment.
void runAcrobotwParamQ(double a, double g, double ee, int i, int d) {
	mt19937_64 generator(0);
	int numTrials = 100, numEpisodes = 100, maxEpisodeLength = 3000;
	double gamma = 1.0;
	Acrobot e;
	//													alpha		gamma	epsilon	iOrder	dOrder
	QLearning a1(e.getStateDim(), e.getNumActions(),	a,g,ee,i,d);
	Sarsa a2(e.getStateDim(), e.getNumActions(),		a,g,ee,i,d);
	vector<double> means1, vars1;
	vector<double> means2(numEpisodes,0.0);
	vector<double> vars2(numEpisodes,0.0);
	runExperiment(a1, e, numTrials, numEpisodes, maxEpisodeLength, gamma, generator, means1, vars1);
	// runExperiment(a2, e, numTrials, numEpisodes, maxEpisodeLength, gamma, generator, means2, vars2);
	ofstream out("../../../output/"+to_string(numEpisodes)+"out_Acrobot-"+to_string(a)+"-"+to_string(g)+"-"+to_string(ee)+"-"+to_string(i)+"-"+to_string(d)+"-"+to_string(means1[numEpisodes-1])+"qlearning.csv");
	out << "Number of Episodes,"
		<< "Q-Learning,Sarsa,"
		<< "Stddev Q-Learning,Stddev Sarsa" << endl;
	for (int epCount = 0; epCount < numEpisodes; epCount++) {
		out << epCount << ","
			<< means1[epCount] << "," << means2[epCount] << "," 
			<< sqrt(vars1[epCount]) << "," << sqrt(vars2[epCount]) << endl;
	}
	out.close();
	cout << to_string(means1[numEpisodes-1]);
}

// See runMountainCar: This is the same thing, but for the Acrobot environment.
void runAcrobotwParamS(double a, double g, double ee, int i, int d) {
	mt19937_64 generator(0);
	int numTrials = 100, numEpisodes = 100, maxEpisodeLength = 3000;
	double gamma = 1.0;
	Acrobot e;
	//													alpha		gamma	epsilon	iOrder	dOrder
	QLearning a1(e.getStateDim(), e.getNumActions(),	a,g,ee,i,d);
	Sarsa a2(e.getStateDim(), e.getNumActions(),		a,g,ee,i,d);
	vector<double> means2, vars2;
	vector<double> means1(numEpisodes,0.0);
	vector<double> vars1(numEpisodes,0.0);
	// runExperiment(a1, e, numTrials, numEpisodes, maxEpisodeLength, gamma, generator, means1, vars1);
	runExperiment(a2, e, numTrials, numEpisodes, maxEpisodeLength, gamma, generator, means2, vars2);
	ofstream out("../../../output/"+to_string(numEpisodes)+"out_Acrobot-"+to_string(a)+"-"+to_string(g)+"-"+to_string(ee)+"-"+to_string(i)+"-"+to_string(d)+"-"+to_string(means2[numEpisodes-1])+"sarsa.csv");
	out << "Number of Episodes,"
		<< "Q-Learning,Sarsa,"
		<< "Stddev Q-Learning,Stddev Sarsa" << endl;
	for (int epCount = 0; epCount < numEpisodes; epCount++) {
		out << epCount << ","
			<< means1[epCount] << "," << means2[epCount] << "," 
			<< sqrt(vars1[epCount]) << "," << sqrt(vars2[epCount]) << endl;
	}
	out.close();
	cout << to_string(means2[numEpisodes-1]);
}

// See runMountainCar: This is the same thing, but for the Gridworld environment.
void runGridworld() {
	mt19937_64 generator(0);
	int numTrials = 100, numEpisodes = 20, maxEpisodeLength = 1000;
	double gamma = 1.0;
	Gridworld e;
	//													alpha		gamma	epsilon	iOrder	dOrder
	QLearning a1(e.getStateDim(), e.getNumActions(),	0.01,		1,		0.1,	2,		0);
	Sarsa a2(e.getStateDim(), e.getNumActions(),		70,			1,		0.95,	1,		0);
	// HINT: Above, do not change iOrder and dOrder. These settings, combined with how the Gridworld is implemented,
	// result in the agents using a tabular representation, which is great for Gridworlds!
	vector<double> means1, vars1, means2, vars2;
	runExperiment(a1, e, numTrials, numEpisodes, maxEpisodeLength, gamma, generator, means1, vars1);
	runExperiment(a2, e, numTrials, numEpisodes, maxEpisodeLength, gamma, generator, means2, vars2);
	ofstream out("../../../output/out_Gridworld.csv");
	out << "Number of Episodes,"
		<< "Q-Learning,Sarsa,"
		<< "Stddev Q-Learning,Stddev Sarsa" << endl;
	for (int epCount = 0; epCount < numEpisodes; epCount++) {
		out << epCount << ","
			<< means1[epCount] << "," << means2[epCount] << "," 
			<< sqrt(vars1[epCount]) << "," << sqrt(vars2[epCount]) << endl;
	}
	out.close();
}

// See runMountainCar: This is the same thing, but for the Gridworld environment.
void runGridworldwParamQ(double a, double g, double ee, int i, int d) {
	mt19937_64 generator(0);
	int numTrials = 100, numEpisodes = 20, maxEpisodeLength = 1000;
	double gamma = 1.0;
	Gridworld e;
	//													alpha		gamma	epsilon	iOrder	dOrder
	QLearning a1(e.getStateDim(), e.getNumActions(),	a,g,ee,i,d);
	Sarsa a2(e.getStateDim(), e.getNumActions(),		a,g,ee,i,d);
	// HINT: Above, do not change iOrder and dOrder. These settings, combined with how the Gridworld is implemented,
	// result in the agents using a tabular representation, which is great for Gridworlds!
	vector<double> means1, vars1;
	vector<double> means2(numEpisodes,0.0);
	vector<double> vars2(numEpisodes,0.0);
	// Disable QLearning
	runExperiment(a1, e, numTrials, numEpisodes, maxEpisodeLength, gamma, generator, means1, vars1);
	// runExperiment(a2, e, numTrials, numEpisodes, maxEpisodeLength, gamma, generator, means2, vars2);
	printf("Writing csv...");
	ofstream out("../../../output/"+to_string(numEpisodes)+"out_Gridworld-"+to_string(a)+"-"+to_string(g)+"-"+to_string(ee)+"-"+to_string(i)+"-"+to_string(d)+"-"+to_string(means1[numEpisodes-1])+"qlearning.csv");
	out << "Number of Episodes,"
		<< "Q-Learning,Sarsa,"
		<< "Stddev Q-Learning,Stddev Sarsa" << endl;
	for (int epCount = 0; epCount < numEpisodes; epCount++) {
		out << epCount << ","
			<< means1[epCount] << "," << means2[epCount] << "," 
			<< sqrt(vars1[epCount]) << "," << sqrt(vars2[epCount]) << endl;
	}
	out.close();
	cout << to_string(means1[numEpisodes-1]);
}
// See runMountainCar: This is the same thing, but for the Gridworld environment.
void runGridworldwParamS(double a, double g, double ee, int i, int d) {
	mt19937_64 generator(0);
	int numTrials = 100, numEpisodes = 20, maxEpisodeLength = 1000;
	double gamma = 1.0;
	Gridworld e;
	//													alpha		gamma	epsilon	iOrder	dOrder
	QLearning a1(e.getStateDim(), e.getNumActions(),	a,g,ee,i,d);
	Sarsa a2(e.getStateDim(), e.getNumActions(),		a,g,ee,i,d);
	// HINT: Above, do not change iOrder and dOrder. These settings, combined with how the Gridworld is implemented,
	// result in the agents using a tabular representation, which is great for Gridworlds!
	vector<double> means2, vars2;
	vector<double> means1(numEpisodes,0.0);
	vector<double> vars1(numEpisodes,0.0);
	// Disable QLearning
	// runExperiment(a1, e, numTrials, numEpisodes, maxEpisodeLength, gamma, generator, means1, vars1);
	runExperiment(a2, e, numTrials, numEpisodes, maxEpisodeLength, gamma, generator, means2, vars2);
	printf("Writing csv...");
	ofstream out("../../../output/"+to_string(numEpisodes)+"out_Gridworld-"+to_string(a)+"-"+to_string(g)+"-"+to_string(ee)+"-"+to_string(i)+"-"+to_string(d)+"-"+to_string(means2[numEpisodes-1])+"sarsa.csv");
	out << "Number of Episodes,"
		<< "Q-Learning,Sarsa,"
		<< "Stddev Q-Learning,Stddev Sarsa" << endl;
	for (int epCount = 0; epCount < numEpisodes; epCount++) {
		out << epCount << ","
			<< means1[epCount] << "," << means2[epCount] << "," 
			<< sqrt(vars1[epCount]) << "," << sqrt(vars2[epCount]) << endl;
	}
	out.close();
	cout << to_string(means2[numEpisodes-1]);
}

// Entry point for the program. We won't use the arguments this time.
int main(int argc, char * argv[])
{
	// cout << "Starting Mountain Car runs..." << endl;
	// runMountainCar();	// Run the mountain car experiments (see the function above). The lines below are similar, but for other MDPs.
	// cout << "\tDone.\nStarting Cart Pole runs..." << endl;
	// runCartPole();
	// cout << "\tDone.\nStarting Acrobot runs..." << endl;
	// runAcrobot();
	// cout << "\tDone.\nStarting Gridworld runs..." << endl;
	// runGridworld();

	// 0.00001, 0.001, 0.1, 1, 10
	// vector<double> as = vector<double>{0.00001, 0.001, 0.1, 1, 10};
	// vector<double> gs = vector<double>{1.0};
	// vector<double> es = vector<double>{0.0, 0.1, 0.2, 0.5, 0.9};
	// vector<int> is = vector<int>{1,2,3,4,5};
	// vector<int> ds = vector<int>{0,1,2};
	//0.060000-1.000000-0.000000-2-1-1011.000000
	//0.020000-1.000000-0.100000-2-0--141.620000

	//vector<double> as = vector<double>{ 0.05 };
	//vector<double> gs = vector<double>{ 1.0 };
	//vector<double> es = vector<double>{ 0.0005 };
	//vector<int> is = vector<int>{ 1 };
	//vector<int> ds = vector<int>{ 1 };
	//0.010000-1.000000-0.100000-2-0--144.910000
	//0.060000-1.000000-0.000000-2-1-1011.000000
	//40out_Mountain-0.040000-1.000000-0.100000-1-1--166.070000qlearning
	//50out_CartPole-0.008000-1.000000-0.100000-2-2-904.580000sarsa
	vector<double> as = vector<double>{0.001};
	vector<double> gs = vector<double>{1.0};
	vector<double> es = vector<double>{0.1};
	vector<int> is = vector<int>{3};
	vector<int> ds = vector<int>{0};
	for (double a : as) {
		for (double g : gs) {
			for (double ee : es) {
				for (int i : is) {
					for (int d : ds) {
						string run = "out-"+to_string(a)+"-"+to_string(g)+"-"+to_string(ee)+"-"+to_string(i)+"-"+to_string(d);
						cout << endl << run << endl;
						runGridworldwParamQ(a,g,ee,i,d);
					}
				}
			}
		}
	}
	
}


/* Acrobat */

/*Q Learning */
/*
Grid search space: 
-----------------

alpha = {0.01, 0.001, 0.0001};
gamma = {1.0};
epsilon = {0.3, 0.4};
iOrder = {3, 4, 5};
dOrder = {0};


int numTrials = 100, numEpisodes = 100, maxEpisodeLength = 3000;

Actual run:
-----------
alpha = {0.01, 0.001};
gamma = {1.0};
epsilon = {0.3, 0.4};
iOrder = {4, 5};
dOrder = {0};

int numTrials = 100, numEpisodes = 100, maxEpisodeLength = 3000;

Optimal hyperparameters:
-----------------------

alpha = 0.001;
gamma = 1.0;
epsilon = 0.3;
iOrder = 4;
dOrder = 0;


int numTrials = 100, numEpisodes = 100, maxEpisodeLength = 3000;

Optimal value = -6.732000


*/

/* Sarsa */

/*
Grid search space:
-----------------

alpha = {0.01, 0.02, 0.03};
gamma = {1.0};
epsilon = {0.04, 0.05, 0.06};
iOrder = {3, 4};
dOrder = {0};


int numTrials = 100, numEpisodes = 100, maxEpisodeLength = 3000;

Actual run:
-----------
alpha = {0.01, 0.02, 0.03};
gamma = {1.0};
epsilon = {0.04, 0.05, 0.06};
iOrder = {3, 4};
dOrder = {0};

int numTrials = 100, numEpisodes = 100, maxEpisodeLength = 3000;

Optimal hyperparameters:
-----------------------

alpha = 0.001;
gamma = 1.0;
epsilon = 0.04;
iOrder = 4;
dOrder = 0;


int numTrials = 100, numEpisodes = 100, maxEpisodeLength = 3000;

Optimal value = -3.853000


*/

/* Cartpole */

/*Q Learning */
/*
Grid search space:
-----------------

alpha = {0.01, 0.001, 0.0001};
gamma = {1.0};
epsilon = {0.3, 0.4};
iOrder = {3, 4, 5};
dOrder = {0};


int numTrials = 50, numEpisodes = 50, maxEpisodeLength = INT_MAX;

Actual run:
-----------
alpha = {0.01, 0.001};
gamma = {1.0};
epsilon = {0.3, 0.4};
iOrder = {4, 5};
dOrder = {0};

int numTrials = 50, numEpisodes = 50, maxEpisodeLength = INT_MAX;

Optimal hyperparameters:
-----------------------

alpha = ;
gamma = 1.0;
epsilon = ;
iOrder = ;
dOrder = 0;


int numTrials = 50, numEpisodes = 50, maxEpisodeLength = INT_MAX;


*/

/* Sarsa */

/* Mountain Car */

/*Q Learning */
/*
Grid search space:
-----------------

alpha = {0.01, 0.02, 0.03};
gamma = {1.0};
epsilon = {0.1, 0.2, 0.3};
iOrder = {1, 2};
dOrder = {1};

int numTrials = 100, numEpisodes = 40, maxEpisodeLength = 20000

Actual run:
-----------
alpha = {0.01, 0.02, 0.03};
gamma = {1.0};
epsilon = {0.1, 0.2, 0.3};
iOrder = {1, 2};
dOrder = {1};

int numTrials = 100, numEpisodes = 40, maxEpisodeLength = 20000

Optimal hyperparameters:
-----------------------

alpha = 0.02;
gamma = 1.0;
epsilon = 0.1;
iOrder = 1;
dOrder = 1;

int numTrials = 100, numEpisodes = 40, maxEpisodeLength = 20000

Optimal value: -141.42


*/

/* Sarsa */

/*

alpha = {0.01, 0.02, 0.03};
gamma = {1.0};
epsilon = {0.1, 0.2, 0.3};
iOrder = {1, 2};
dOrder = {1};

int numTrials = 100, numEpisodes = 40, maxEpisodeLength = 20000

Actual run:
-----------
alpha = {0.01, 0.02, 0.03};
gamma = {1.0};
epsilon = {0.1, 0.2, 0.3};
iOrder = {1, 2};
dOrder = {1};

int numTrials = 100, numEpisodes = 40, maxEpisodeLength = 20000

Optimal hyperparameters:
-----------------------

alpha = 0.02;
gamma = 1.0;
epsilon = 0.1;
iOrder = 1;
dOrder = 1;

int numTrials = 100, numEpisodes = 40, maxEpisodeLength = 20000

Optimal value: -145.53
*/

/* Gridworld */

/*Q Learning */
/*
Grid search space:
-----------------

alpha = {0.01, 0.02, 0.03};
gamma = {1.0};
epsilon = {0.1, 0.2};
iOrder = {1, 2, 3};
dOrder = {0};


int numTrials = 100, numEpisodes = 20, maxEpisodeLength = 1000;

Actual run:
-----------
alpha = {0.01, 0.02};
gamma = {1.0};
epsilon = {0.1, 0.2};
iOrder = {2, 3};
dOrder = {0};

int numTrials = 100, numEpisodes = 20, maxEpisodeLength = 1000;

Optimal value = -11.43

Optimal hyperparameters:
-----------------------

alpha = ;
gamma = 1.0;
epsilon = ;
iOrder = ;
dOrder = 0;


int numTrials = 100, numEpisodes = 20, maxEpisodeLength = 1000;

*/

/* Sarsa */

/*

Grid search space:
-----------------

alpha = {0.01, 0.02, 0.03};
gamma = {1.0};
epsilon = {0.1, 0.2};
iOrder = {1, 2, 3};
dOrder = {0};


int numTrials = 100, numEpisodes = 20, maxEpisodeLength = 1000;

Actual run:
-----------
alpha = {0.01, 0.02, 0.03};
gamma = {1.0};
epsilon = {0.01, 0.02, 0.03};
iOrder = {1, 2};
dOrder = {0};

int numTrials = 100, numEpisodes = 20, maxEpisodeLength = 1000;

Optimal hyperparameters:
-----------------------

alpha = 0.02;
gamma = 1.0;
epsilon = 0.01;
iOrder = 1;
dOrder = 0;


int numTrials = 100, numEpisodes = 20, maxEpisodeLength = 1000;

Optimal value = -8.27

*/
