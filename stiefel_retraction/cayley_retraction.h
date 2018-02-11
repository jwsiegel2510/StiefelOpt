/* Author: Jonathan Siegel

Implements a retraction on the Stiefel manifold using the Cayley transform.
*/

#ifndef _STIEFEL_RETRACTION_CAYLEY_RETRACTION__
#define _STIEFEL_RETRACTION_CAYLEY_RETRACTION__

#include "../Eigen/Dense"
#include "../util/base/vector_ops.h"
#include <vector>

namespace {
	using ::Eigen::MatrixXd;
	using ::Eigen::HouseholderQR;
        using namespace ::util::base::vector_ops;
        using std::vector;
}

namespace stiefel_retraction {
namespace cayley_retraction {

template<class P = vector<vector<double> >, class V = vector<vector<double> > > class CayleyRetraction {
private:
	MatrixXd U;
	MatrixXd VM;
	MatrixXd X;
	MatrixXd Y;
	MatrixXd TV;
public:
        void retract(P& iterate, const V& direction, double dt);
        // Sets temporary_iterate equal to the tangent vector correponding to
        // the increment from iterate to temporary_iterate, sets iterate equal to temporary iterate
        // and sets y_iterate equal to the overshoot of a retraction through iterate and temporary iterate by an amount beta.
        void apply_momentum(P& y_iterate, P& iterate, P& temporary_iterate, double beta);
        double norm_sq(const V& grad, const P& iterate);
	double calculate_BB_step_size(const P& iterate, const P& prev_iterate, const V& grad, const V& prev_grad, int it);
};

template<> void CayleyRetraction<>::retract(vector<vector<double> >& iterate, const vector<vector<double> >& direction, double dt) {
	U.resize(iterate[0].size(), 2 * iterate.size());
	VM.resize(iterate[0].size(), 2 * iterate.size());
	X.resize(iterate[0].size(), iterate.size());
	for (int i = 0; i < iterate.size(); ++i) {
		for (int j = 0; j < iterate[i].size(); ++j) {
			U(j , i) = -0.5 * dt * iterate[i][j];
			VM(j, i + iterate.size()) = iterate[i][j];
			U(j, i + iterate.size()) = -0.5 * dt * direction[i][j];
			VM(j , i) = -1.0 * direction[i][j];
			X(j , i) = iterate[i][j];
		}
	}
	X = X - 2.0 * U * (MatrixXd::Identity(2 * iterate.size(), 2 * iterate.size()) + VM.transpose() * U).colPivHouseholderQr().solve(VM.transpose() * X);
	HouseholderQR<MatrixXd> hh(X);
	X = hh.householderQ() * MatrixXd::Identity(iterate[0].size(), iterate.size());
	for (int i = 0; i < iterate.size(); ++i) {
		for (int j = 0; j < iterate[i].size(); ++j) {
			iterate[i][j] = X(j , i);	
		}
	}
}

template<> double CayleyRetraction<>::calculate_BB_step_size(const vector<vector<double> >& prev_iterate, const vector<vector<double> >& iterate,
                                                        const vector<vector<double> >& grad, const vector<vector<double> >& prev_grad, int it) {
        return (it%2) ? dot(iterate - prev_iterate, iterate - prev_iterate) /
                     dot(iterate - prev_iterate, grad - prev_grad)
                   : dot(iterate - prev_iterate, grad - prev_grad) /
                     dot(grad - prev_grad, grad - prev_grad);
}

template<> double CayleyRetraction<>::norm_sq(const vector<vector<double> >& grad, const vector<vector<double> >& iterate) {
        vector<vector<double> > temp(grad);
        for (int i = 0; i < grad.size(); ++i) {
                for (int j = 0; j < grad.size(); ++j) {
                        temp[i] -= iterate[j] * dot(grad[j], iterate[i]);
                }
        }
        return dot(temp, temp);
}

template<> void CayleyRetraction<>::apply_momentum(vector<vector<double> >& y_iterate, vector<vector<double> >& iterate, vector<vector<double> >& temporary_iterate, double beta) {
	X.resize(iterate[0].size(), iterate.size());
	Y.resize(y_iterate[0].size(), y_iterate.size());
	TV.resize(iterate[0].size(), iterate.size());
	for (int i = 0; i < iterate.size(); ++i) {
		for (int j = 0; j < iterate[i].size(); ++j) {
			X(j , i) = iterate[i][j];
			Y(j , i) = temporary_iterate[i][j];
		}
	}
	y_iterate = iterate;
	iterate = temporary_iterate;
	TV = 2.0 * (MatrixXd::Identity(iterate.size(), iterate.size()) + X.transpose() * Y).colPivHouseholderQr().solve(Y.transpose()).transpose();
	for (int i = 0; i < iterate.size(); ++i) {
		for (int j = 0; j < iterate[i].size(); ++j) {
			temporary_iterate[i][j] = TV(j , i);
		}
	}
	retract(y_iterate, temporary_iterate, 1.0 + beta);
}

}
}

#endif
