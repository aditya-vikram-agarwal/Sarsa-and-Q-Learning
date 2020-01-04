#pragma once

#include "stdafx.h"

// A class implementing the Fourier basis
class FourierBasis
{
public:
	void init(const int & inputDimension, int iOrder, int dOrder);
	int getNumOutputs() const;
	std::vector<double> basify(const std::vector<double> & x) const;

private:
	int nTerms;							// Total number of outputs
	int inputDimension;
	std::vector<std::vector<double>> c;	// Coefficients
};