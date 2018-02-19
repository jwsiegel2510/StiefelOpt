/* Author: Jonathan Siegel

Implements a retraction on the Stiefel manifold using the Cayley transform.
*/

#ifndef _STIEFEL_RETRACTION_CAYLEY_RETRACTION__
#define _STIEFEL_RETRACTION_CAYLEY_RETRACTION__

#include "../Eigen/Dense"
#include <cstdio>

namespace {
	using ::Eigen::MatrixXd;
	using ::Eigen::HouseholderQR;
}

namespace stiefel_retraction {
namespace cayley_retraction {

template<class P = MatrixXd, class V = MatrixXd > class CayleyRetraction {
private:
	MatrixXd U;
	MatrixXd VM;
	MatrixXd X;
public:
        void retract(P& iterate, const V& direction, double dt);
        // Sets temporary_iterate equal to the tangent vector correponding to
        // the increment from iterate to temporary_iterate, sets iterate equal to temporary iterate
        // and sets y_iterate equal to the overshoot of a retraction through iterate and temporary iterate by an amount beta.
        void apply_momentum(P& y_iterate, P& iterate, P& temporary_iterate, double beta);
        double norm_sq(const V& grad, const P& iterate);
	double calculate_BB_step_size(const P& iterate, const P& prev_iterate, const V& grad, const V& prev_grad, int it);
};

template<> void CayleyRetraction<>::retract(MatrixXd& iterate, const MatrixXd& direction, double dt) {
	// For reasons of numerical stability, we must (approximately) project direction ont othe dual space.
	X = direction - 0.5 * iterate * (iterate.transpose() * direction + direction.transpose() * iterate);
	U.resize(iterate.rows(), 2 * iterate.cols());
	VM.resize(iterate.rows(), 2 * iterate.cols());
	U.block(0, 0, iterate.rows(), iterate.cols()) = -0.5 * dt * X;
	U.block(0, iterate.cols(), iterate.rows(), iterate.cols()) = 0.5 * dt * iterate;
	VM.block(0, 0, iterate.rows(), iterate.cols()) = iterate;
	VM.block(0, iterate.cols(), iterate.rows(), iterate.cols()) = X;
	iterate = iterate - 2.0 * U * (MatrixXd::Identity(2 * iterate.cols(), 2 * iterate.cols()) + VM.transpose() * U).colPivHouseholderQr().solve(VM.transpose() * iterate);
	HouseholderQR<MatrixXd> hh(iterate);
	iterate = hh.householderQ() * MatrixXd::Identity(iterate.rows(), iterate.cols());
}

template<> double CayleyRetraction<>::calculate_BB_step_size(const MatrixXd& iterate, const MatrixXd& prev_iterate,
                                                        const MatrixXd& grad, const MatrixXd& prev_grad, int it) {
        return (it%2) ?  ((iterate - prev_iterate).transpose() * (iterate - prev_iterate)).trace() / 
                     ((iterate - prev_iterate).transpose() * (grad - prev_grad)).trace()
                   : ((iterate - prev_iterate).transpose() * (grad - prev_grad)).trace() /
                     ((grad - prev_grad).transpose() * (grad - prev_grad)).trace();
}

template<> double CayleyRetraction<>::norm_sq(const MatrixXd& grad, const MatrixXd& iterate) {
        X.resize(grad.rows(), grad.cols());
	X = grad;
	X -= iterate * (grad.transpose() * iterate);
        return (X.transpose() * X).trace();
}

template<> void CayleyRetraction<>::apply_momentum(MatrixXd& y_iterate, MatrixXd& iterate, MatrixXd& temporary_iterate, double beta) {	
	y_iterate = iterate;
	for (int i = 0; i < iterate.cols(); ++i) {
		if (temporary_iterate.col(i).dot(iterate.col(i)) < 0) temporary_iterate.col(i) *= -1.0;
	}
	iterate = temporary_iterate;
	temporary_iterate = 2.0 * (MatrixXd::Identity(iterate.cols(), iterate.cols()) + iterate.transpose() * y_iterate).colPivHouseholderQr().solve(
		(iterate - y_iterate + .75 * y_iterate * (MatrixXd::Identity(iterate.cols(), iterate.cols()) - y_iterate.transpose() * iterate))
			.transpose()).transpose();
	temporary_iterate += 0.5 * (MatrixXd::Identity(iterate.cols(), iterate.cols()) + y_iterate.transpose() * iterate).colPivHouseholderQr().solve(
		(y_iterate * (MatrixXd::Identity(iterate.cols(), iterate.cols()) - iterate.transpose() * y_iterate)).transpose()).transpose();
	retract(y_iterate, temporary_iterate, 1.0 + beta); 
}

}
}

#endif
