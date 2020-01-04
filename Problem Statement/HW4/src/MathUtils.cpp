#include "stdafx.h"

using namespace std;

// See MathUtils.hpp for descriptions of each of the functions listed here.

void incrementCounter(vector<double> & buff, const int & maxDigit) {
	for (int i = 0; i < (int)buff.size(); i++) {
		buff[i]++;
		if (buff[i] <= maxDigit)
			break;
		buff[i] = 0;
	}
}

int ipow(const int & a, const int & b) {
	if (b == 0) return 1;
	if (b == 1) return a;
	int tmp = ipow(a, b / 2);
	if (b % 2 == 0) return tmp * tmp;
	else return a * tmp * tmp;
}

double dot(const std::vector<double> & x, const std::vector<double> & y) {
	double result = 0;
	for (int i = 0; i < (int)x.size(); i++)
		result += x[i] * y[i];
	return result;
}

double mean(const std::vector<double> & v) {
	double result = 0;
	for (int i = 0; i < (int)v.size(); i++)
		result += v[i];
	return result / (double)v.size();
}

double var(const std::vector<double> & v) {
	double mu = mean(v), result = 0;
	for (int i = 0; i < (int)v.size(); i++)
		result += (v[i] - mu)*(v[i] - mu);
	return result / (double)(v.size() - 1);
}

/*
Floating-point modulo:
The result (the remainder) has same sign as the divisor.
Similar to matlab's mod(); Not similar to fmod() because:
Mod(-3,4)= 1
fmod(-3,4)= -3
*/
double Mod(const double & x, const double & y) {
	if (0. == y) return x;
	double m = x - y * std::floor(x / y);
	// handle boundary cases resulted from floating-point cut off:
	if (y > 0) {
		if (m >= y)
			return 0;
		if (m < 0) {
			if (y + m == y) return 0;
			else return (y + m);
		}
	}
	else
	{
		if (m <= y) return 0;
		if (m > 0) {
			if (y + m == y) return 0;
			else return (y + m);
		}
	}
	return m;
}

// wrap [rad] angle to [-PI..PI)
double wrapPosNegPI(const double & theta) {
	return Mod((double)theta + M_PI, (double)2.0*M_PI) - (double)M_PI;
}

// wrap [rad] angle to [0..TWO_PI)
double wrapTwoPI(const double & theta) {
	return Mod((double)theta, (double)(2.0*M_PI));
}

double sign(const double & x) {
	return (x > 0) - (x < 0);
}

double bound(const double & x, const double & minValue, const double & maxValue) {
	return min(maxValue, max(minValue, x));
}

int bound(const int & x, const int & minValue, const int & maxValue) {
	return min(maxValue, max(minValue, x));
}

double normalize(const double & x, const double & minValue, const double & maxValue) {
	double temp = bound(x, minValue, maxValue);
	return (x - minValue) / (maxValue - minValue);
}