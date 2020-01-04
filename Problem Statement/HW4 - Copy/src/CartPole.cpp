#include <stdafx.h>

using namespace std;

CartPole::CartPole() {
	mt19937_64 generator(0);
	newEpisode(generator);
}

int CartPole::getStateDim() const {
	return 4;
}

int CartPole::getNumActions() const {
	return 2;
}

double CartPole::update(const int & action, mt19937_64 & generator) {
	double F = action*uMax + (action - 1)*uMax, omegaDot, vDot, subDt = dt / (double)simSteps;
	for (int i = 0; i < simSteps; i++) {
		omegaDot = (g*sin(theta) + cos(theta)*(muc*sign(v) - F - m*l*omega*omega*sin(theta)) / (m + mc) - mup*omega / (m*l)) / (l*(4.0 / 3.0 - m / (m + mc)*cos(theta)*cos(theta)));
		vDot = (F + m*l*(omega*omega*sin(theta) - omegaDot*cos(theta)) - muc*sign(v)) / (m + mc);
		theta += subDt*omega;
		omega += subDt*omegaDot;
		x += subDt*v;
		v += subDt*vDot;
		theta = wrapPosNegPI(theta);
		t += subDt;
	}
	x = bound(x, xMin, xMax);
	v = bound(v, vMin, vMax);
	theta = bound(theta, thetaMin, thetaMax);
	omega = bound(omega, omegaMin, omegaMax);
	return 1;
}

vector<double> CartPole::getState(mt19937_64 & generator) {
	vector<double> result(4);
	result[0] = normalize(x, xMin, xMax);
	result[1] = normalize(v, vMin, vMax);
	result[2] = normalize(theta, thetaMin, thetaMax);
	result[3] = normalize(omega, omegaMin, omegaMax);
	return result;
}

bool CartPole::inTerminalState() const {
	return ((fabs(theta) > M_PI / 15.0) || (fabs(x) >= 2.4) || (t >= 20.0 + 10 * dt));
}

void CartPole::newEpisode(mt19937_64 & generator) {
	theta = omega = v = x = t = 0;
}