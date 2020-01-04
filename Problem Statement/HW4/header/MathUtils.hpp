#pragma once

#include <stdafx.h>

// Useful math functions. See MathUtils.cpp for implementations

// Increment a counter in base maxDigit+1
void incrementCounter(std::vector<double> & buff, const int & maxDigit);

// Power function for integers
int ipow(const int & a, const int & b);

// Compute the dot-product of two std::vector<double>
double dot(const std::vector<double> & x, const std::vector<double> & y);

// Compute the sample mean of an std::vector<double>
double mean(const std::vector<double> & v);

// Compute the sample variance of an std::vector<double>
double var(const std::vector<double> & v);

/*
Floating-point modulo:
The result (the remainder) has same sign as the divisor.
Similar to matlab's mod(); Not similar to fmod() because:
Mod(-3,4)= 1
fmod(-3,4)= -3
*/
double Mod(const double & x, const double & y);

// wrap [rad] angle to [-PI..PI)
double wrapPosNegPI(const double & theta);	// wrap [rad] angle to [-PI..PI)

// wrap [rad] angle to [0..TWO_PI)
double wrapTwoPI(const double & theta);		// wrap [rad] angle to [0..TWO_PI)

// Get the sign of a double
double sign(const double & x);

// Return x, but bounded to be within [minValue,maxValue]
double bound(const double & x, const double & minValue, const double & maxValue);

// Return x, but bounded to be within [minValue,maxValue]
int bound(const int & x, const int & minValue, const int & maxValue);

// Normalize x to be in the range [0,1], where originally x is in [minValue, maxValue].
double normalize(const double & x, const double & minValue, const double & maxValue);