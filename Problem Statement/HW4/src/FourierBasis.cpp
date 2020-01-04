#include "stdafx.h"

using namespace std;

void FourierBasis::init(const int & inputDimension, int iOrder, int dOrder) {
	this->inputDimension = inputDimension;					// Copy over the provided arguments
	// Compute the total number of terms
	int iTerms = iOrder*inputDimension;						// Number of independent terms
	int dTerms = ipow(dOrder + 1, inputDimension);			// Number of dependent terms
	int oTerms = min(iOrder, dOrder)*inputDimension;		// Overlap of iTerms and dTerms
	nTerms = iTerms + dTerms - oTerms;
	// Initialize c
	c.resize(nTerms);
	vector<double> counter(inputDimension, 0.0);
	int termCount = 0;
	for (; termCount < dTerms; termCount++) {				// First add the dependent terms
		c[termCount] = counter;
		incrementCounter(counter, dOrder);
	}
	for (int i = 0; i < inputDimension; i++) {				// Add the independent terms
		for (int j = dOrder + 1; j <= iOrder; j++) {
			c[termCount] = vector<double>(inputDimension, 0.0);
			c[termCount][i] = (double)j;
			termCount++;
		}
	}
}

int FourierBasis::getNumOutputs() const {
	return nTerms;
}

vector<double> FourierBasis::basify(const vector<double> & x) const {
	vector<double> result(nTerms);
	for (int i = 0; i < nTerms; i++)
		result[i] = cos(M_PI*dot(c[i], x));
	return result;
}