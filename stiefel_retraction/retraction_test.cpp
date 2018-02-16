/* Author: Jonathan Siegel

Tests the Cayley retraction class to ensure that it defines a valid retraction
and solves the iterpolation problem correctly.
*/

#include "cayley_retraction.h"
#include "../Eigen/Dense"
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iostream>

namespace {
	using ::Eigen::MatrixXd;
	using ::Eigen::HouseholderQR;
	using ::Eigen::IOFormat;
	using ::stiefel_retraction::cayley_retraction::CayleyRetraction;
}

int main() {
	srand(time(NULL)); // initialize random seed.
	const IOFormat fmt(2, 1, "\t", " ", "", "", "", "");
	MatrixXd P1(20, 5);
	MatrixXd V(20, 5);
	MatrixXd P2(20, 5);
	for (int i = 0; i < 20; ++i) { // initialize all points randomly
		for (int j = 0; j < 5; ++j) {
			double theta = 2 * M_PI * ((double) rand()) / (RAND_MAX);
			double r = ((double) rand()) / (RAND_MAX);
			P1(i,j) = sqrt(-log(r)) * sin(theta);
                        theta = 2 * M_PI * ((double) rand()) / (RAND_MAX);
                        r = ((double) rand()) / (RAND_MAX);
                        P2(i,j) = sqrt(-log(r)) * sin(theta);
                        theta = 2 * M_PI * ((double) rand()) / (RAND_MAX);
                        r = ((double) rand()) / (RAND_MAX);
                        V(i,j) = sqrt(-log(r)) * sin(theta);			
		}
	}
	HouseholderQR<MatrixXd> h1(P1);
	HouseholderQR<MatrixXd> h2(P2);
	P1 = h1.householderQ() * MatrixXd::Identity(P1.rows(), P1.cols());
	P2 = h2.householderQ() * MatrixXd::Identity(P2.rows(), P2.cols());
	CayleyRetraction<> retraction;
	MatrixXd P = P1; // Test to make sure that the retraction has the correct gradient empirically.
	retraction.retract(P, V, .0001);
	MatrixXd empV = (P - P1) / .0001;
	MatrixXd trueV = V - P1 * (V.transpose() * P1);
	std::cout << (empV - trueV).format(fmt) << "\n";
	printf("Empirical Gradient Error: %lf \n", ((empV - trueV).transpose() * (empV - trueV)).trace());
	MatrixXd Q = P2;
	retraction.apply_momentum(P, P1, Q, 0.0);
	printf("Retraction Error for X: %lf \n", ((P1 - P2).transpose() * (P1 - P2)).trace());
	printf("Retraction Error for Y: %lf \n", ((P - P2).transpose() * (P - P2)).trace());
}
