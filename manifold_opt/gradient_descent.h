/* Author: Jonathan Siegel

Contains a templated implementation of gradient descent on manifolds. 
Passed as template parameters are a class evaluating the gradient and objective and a class
performing the desired retraction. The class performing the retraction is
assumed to define all operations which depend upon the geometry of the manifold, such as
calculating the BB step size (which in the manifold setting requires care and is dependent
on the geometry) and calculating norms of tangent vectors.

The algorithm uses a non-monotone armijo stepsize condition as detailed in

Zhang, H. and Hager, W.W., 2004. A nonmonotone line search technique and its application to unconstrained optimization. 
SIAM journal on Optimization, 14(4), pp.1043-1056.
*/

#ifndef _MANIFOLD_OPT_GRADIENT_DESCENT__
#define _MANIFOLD_OPT_GRADIENT_DESCENT__

#include<cmath>
#include<cstdio>

namespace manifold_opt {
namespace gradient_descent {

template<template<class, class> class Evaluator, template<class, class> class Retraction, class P, class V>
void descent_opt(P& iterate, Evaluator<P, V> evaluator, Retraction<P, V> retraction, double gtol = 1e-4) {
        const static double rho = 1e-4; // Parameter for the Armijo condition.
        const static double gamma = .5; // Parameter controlling the non-monotone step size.
        const static double eta = .2; // Parameter determining the factor by which the step decreases if Armijo condition isn't met.
        double step_size = 1e-1; // Initial step size.
        double Q = 1; // updated parameter from non-monotone stepping scheme.
        double Cval = evaluator.evaluate(iterate); // set initial non-monotone target equal to the objective
	
	// The data required to calcualte the BB step size, retraction, and stopping condition.
	P temporary_iterate(iterate);
	V grad;
	V prev_grad;
	
	int it = 0;
	while(1) {
                // Evaluate gradient.
                evaluator.evaluate_grad(grad, temporary_iterate);

		// Calculate the squared norm of the gradient and check stopping condition.
		double grad_norm_sq = retraction.norm_sq(grad, temporary_iterate);
		if (sqrt(grad_norm_sq) < gtol) {
			iterate = temporary_iterate; return;
		}

		// Calculate BB step size. This is done entirely by the class responsible for the retraction.
		// It is passed as imput the previous and current iterates and the previous and current gradients.
		if (it > 0) 
			step_size = retraction.calculate_BB_step_size(iterate, temporary_iterate, grad, prev_grad, it);
		iterate = temporary_iterate;
		
                // Calculate retraction. Decrease step size until non-monotone Armijo condition is satisfied.
                bool done = false;
                while (!done) {
			retraction.retract(temporary_iterate, -1.0 * grad, step_size);
			double F = evaluator.evaluate(temporary_iterate);
                        if (F <= Cval - rho * step_size * grad_norm_sq) {
                                done = true;
                                Cval = (gamma * Q * Cval + F) / (gamma * Q + 1);
                                Q = gamma * Q + 1;
                                ++it;
                                prev_grad = grad;
                        } else {
                                temporary_iterate = iterate;
                                step_size *= eta;
                        }
		}
	}
}

}
}
#endif
