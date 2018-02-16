/* Author: Jonathan Siegel

Implements a retraction on the Stiefel manifold using the Cayley transform.
*/

#ifndef _STIEFEL_RETRACTION_CAYLEY_RETRACTION__
#define _STIEFEL_RETRACTION_CAYLEY_RETRACTION__

#include "../Eigen/Dense"

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
	U.resize(iterate.rows(), 2 * iterate.cols());
	VM.resize(iterate.rows(), 2 * iterate.cols());
	U.block(0, 0, iterate.rows(), iterate.cols()) = -0.5 * dt * direction;
	U.block(0, iterate.cols(), iterate.rows(), iterate.cols()) = 0.5 * dt * iterate;
	VM.block(0, 0, iterate.rows(), iterate.cols()) = iterate;
	VM.block(0, iterate.cols(), iterate.rows(), iterate.cols()) = direction;
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
	/* y_iterate = iterate;
	iterate = temporary_iterate;
	temporary_iterate = 2.0 * (MatrixXd::Identity(iterate.cols(), iterate.cols()) + y_iterate.transpose() * iterate).colPivHouseholderQr().solve(iterate.transpose()).transpose();
	retract(y_iterate, temporary_iterate, 1.0 + beta); */
	y_iterate = (1.0 + beta) * temporary_iterate - beta * iterate;
        HouseholderQR<MatrixXd> hh(y_iterate);
        y_iterate = hh.householderQ() * MatrixXd::Identity(y_iterate.rows(), y_iterate.cols());
	iterate = temporary_iterate;
}

}
}

#endif
