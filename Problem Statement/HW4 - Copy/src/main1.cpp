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

// See runMountainCar: This is the same thing, but for the Gridworld environment.
void runGridworld() {
	mt19937_64 generator(0);
	int numTrials = 100, numEpisodes = 20, maxEpisodeLength = 1000;
	double gamma = 1.0;
	Gridworld e;
	//													alpha		gamma	epsilon	iOrder	dOrder
	QLearning a1(e.getStateDim(), e.getNumActions(),	0.0001,	1,		0.05,		1,		0);
	Sarsa a2(e.getStateDim(), e.getNumActions(),		0.0001,			1,		0.05,	1,		0);
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

// Entry point for the program. We won't use the arguments this time.
int main(int argc, char * argv[])
{
	cout << "Starting Mountain Car runs..." << endl;
	//runMountainCar();	// Run the mountain car experiments (see the function above). The lines below are similar, but for other MDPs.
	cout << "\tDone.\nStarting Cart Pole runs..." << endl;
	//runCartPole();
	cout << "\tDone.\nStarting Acrobot runs..." << endl;
	//runAcrobot();
	cout << "\tDone.\nStarting Gridworld runs..." << endl;
	runGridworld();
}