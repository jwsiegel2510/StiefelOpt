/* Author: Jonathan Siegel

Tests the accelerated gradient descent with adaptive restart procedure on the eigenvalue problem.
*/

#include "manifold_opt/accel_grad_descent.h"
#include "stiefel_retraction/cayley_retraction.h"
#include "Eigen/Dense"
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>

namespace {
	using ::manifold_opt::accel_grad_descent::accel_opt;
	using ::stiefel_retraction::cayley_retraction::CayleyRetraction;
	using ::Eigen::MatrixXd;
        using ::Eigen::HouseholderQR;
}

template<class P = MatrixXd, class V = MatrixXd> class EigenvectorObjective {
public:
	EigenvectorObjective() {}

	double evaluate(const P& input) {
		double ans = 0;
		for (int i = 0; i < input.rows(); ++i) {
			for (int j = 0; j < input.cols(); ++j) {
				ans += i * input(i,j) * input(i,j);
			}
		}
		return ans;
	}

	double evaluate_grad(V& grad, const P& input) {
		grad.resize(input.rows(), input.cols());
                for (int i = 0; i < input.rows(); ++i) {
                        for (int j = 0; j < input.cols(); ++j) {
                                grad(i,j) = i * input(i,j);
                        }
                }
	}
};

int main() {
	int k;
	printf("Input the number of eigenvectors to be calculated.\n");
	scanf("%d", &k);
	srand(time(NULL)); // initialize random seed.
	EigenvectorObjective<> evaluator;
	MatrixXd iterate;
	for (int i = 100; i < 1000; ++i) {
		iterate.resize(i + 1, k); // Initialize iterate to be a uniformly random point on the Stiefel manifold.
		for (int j = 0; j < i + 1; ++j) {
			for (int l = 0; l < k; ++l) {
				double theta = 2 * M_PI * ((double) rand()) / (RAND_MAX);
				double r = ((double) rand()) / (RAND_MAX);
				iterate(j,l) = sqrt(-log(r)) * sin(theta);
			}
		}
		HouseholderQR<MatrixXd> hh(iterate);
		iterate = hh.householderQ() * MatrixXd::Identity(iterate[0].size(), iterate.size());
		printf("%d %d ", i, accel_opt(iterate, evaluator, CayleyRetraction<>(), 1e-3, i + 1));
	}
}