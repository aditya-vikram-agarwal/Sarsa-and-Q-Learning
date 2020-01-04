#include <stdafx.h>

using namespace std;

Acrobot::Acrobot() {
	mt19937_64 generator(0);
	newEpisode(generator);
}

int Acrobot::getStateDim() const {
	return 4;
}

int Acrobot::getNumActions() const {
	return 3;
}

double Acrobot::update(const int & action, mt19937_64 & generator) {
	double u = (double)(action - 1)*fmax, h = dt / integShritte;
	hilf[0] = theta1;
	hilf[1] = theta2;
	hilf[2] = theta1Dot;
	hilf[3] = theta2Dot;
	for (int i = 0; i<integShritte; i++) {
		f(hilf, u, s0_dot);
		for (int j = 0; j<4; j++)
			s1[j] = hilf[j] + (h / 2)*s0_dot[j];
		f(s1, u, s1_dot);
		for (int j = 0; j<4; j++)
			s2[j] = hilf[j] + (h / 2)*s1_dot[j];
		f(s2, u, s2_dot);
		for (int j = 0; j<4; j++)
			s3[j] = hilf[j] + h * s2_dot[j];
		f(s3, u, s3_dot);
		for (int j = 0; j<4; j++)
			hilf[j] = hilf[j] + (h / 6) * (s0_dot[j] + 2 * (s1_dot[j] + s2_dot[j]) + s3_dot[j]);
	}
	for (int j = 0; j<4; j++)
		ss[j] = hilf[j];
	if (ss[0] > M_PI)
		ss[0] -= 2 * M_PI;
	if (ss[0] < -M_PI)
		ss[0] += 2 * M_PI;
	if (ss[1] > M_PI)
		ss[1] -= 2 * M_PI;
	if (ss[1] < -M_PI)
		ss[1] += 2 * M_PI;
	theta1 = ss[0];
	theta2 = ss[1];
	theta1Dot = ss[2];
	theta2Dot = ss[3];

	// Enforce joint angle constraints
	theta1 = wrapPosNegPI(theta1);
	theta2 = wrapPosNegPI(theta2);

	// Enforce joint angle derivative constraints
	theta1Dot = bound(theta1Dot, -4 * M_PI, 4 * M_PI);
	theta2Dot = bound(theta2Dot, -9 * M_PI, 9 * M_PI);

	t += dt;

	if (inTerminalState())
		return 10;
	return -.1;
}

vector<double> Acrobot::getState(mt19937_64 & generator) {
	vector<double> result(4);
	result[0] = normalize(theta1, -M_PI, M_PI);
	result[1] = normalize(theta2, -M_PI, M_PI);
	result[2] = normalize(theta1Dot, -4.0*M_PI, 4.0*M_PI);
	result[3] = normalize(theta2Dot, -9.0*M_PI, 9.0*M_PI);
	return result;
}

bool Acrobot::inTerminalState() const {
	double elbowY = -l1 * cos(theta1);
	double handY = elbowY - l2 * cos(theta1 + theta2);
	return handY > l1;
}

void Acrobot::newEpisode(mt19937_64 & generator) {
	t = theta1 = theta2 = theta1Dot = theta2Dot = 0;
}

// Helper function used by the Runge-Kutta approximation
void Acrobot::f(double s[4], double tau, double * buff) {
	double phi1, phi2, d1, d2, newa1, newa2;
	d1 = m1 * lc1*lc1 + m2 * (l1*l1 + lc2 * lc2 + 2 * l1*lc2*cos(s[1])) + i1 + i2;
	d2 = m2 * (lc2*lc2 + l1 * lc2*cos(s[1])) + i2;
	phi2 = (m2*lc2*g*cos(s[0] + s[1] - M_PI / 2.0));
	phi1 = (-m2 * l1*lc2*s[3] * s[3] * sin(s[1]) - 2 * m2*l1*lc2*s[3] * s[2] * sin(s[1]) + (m1*lc1 + m2 * l1)*g*cos(s[0] - M_PI / 2.0) + phi2);
	newa2 = ((1.0 / (m2*lc2*lc2 + i2 - (d2*d2) / d1)) * (tau + (d2 / d1)*phi1 - m2 * l1*lc2*s[2] * s[2] * sin(s[1]) - phi2));
	newa1 = ((-1.0 / d1) * (d2*newa2 + phi1));
	buff[0] = s[2];
	buff[1] = s[3];
	buff[2] = newa1;
	buff[3] = newa2;
}