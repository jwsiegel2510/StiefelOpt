/* Author: Jonathan Siegel

Contains a templated implementation of the accelerated gradient descent algorithm on manifolds.
Passed as template parameters are a class evaluating the gradient and a class performing the
retraction. The class performing the retraction must also contain a function for calculating the
norm of gradients and a function for performing the extrapolation step in the momentum method.

The functions return the number of iterations required to achieve convergence.
*/

#ifndef _MANIFOLD_OPT_ACCEL_GRAD_DESCENT__
#define _MANIFOLD_OPT_ACCEL_GRAD_DESCENT__

#include<cstdio>
#include<vector>

namespace manifold_opt {
namespace accel_grad_descent {

void Print(const std::vector<std::vector<double> >& V) {
	for (int i = 0; i < V.size(); ++i) {
		for (int j = 0; j < V[i].size(); ++j) {
			printf("%lf ", V[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}

// If information on the smoothness is known, it can be passed as a parameter.
template<template<class, class> class Evaluator, template<class, class> class Retraction, class P, class V>
int accel_opt(P& iterate, Evaluator<P, V> evaluator, Retraction<P, V> retraction, double gtol = 1e-4, double smoothness = 0) {
	bool smoothness_known = (smoothness != 0);
        const static double restart_rho = 1e-2; // Sufficient decrease parameter for restart.
        const static double rho = 5e-1; // Parameter for the Armijo condition.
        const static double eta = .5; // Parameter determining the factor by which the step decreases/increases during armijo search.
        double period_tol = 1.5; // Parameter controlling the tolerance given to the period increase upon decrease of the step size.
				 // Only used if smoothness is not passed.

	P y_iterate(iterate);
	P temporary_iterate;
	V grad;

	double Fval;
	double step_size = .1;
        bool increase = true; // Allow the step size to increase at the first iteration (if smoothness is not passed).
        bool dec_step_size = false; // Decrease the step size to see if the period increases by a large factor (if smoothness not passed). 
        int prev_period;

	int it = 1;
	int k = 0;
	while (1) {
                Fval = evaluator.evaluate(y_iterate);
                evaluator.evaluate_grad(grad, y_iterate);

                // Calculate the squared norm of the gradient and check stopping condition.
                double grad_norm_sq = retraction.norm_sq(grad, y_iterate);

		// Calculate retraction. Decrease step size until armijo condition is met.
		temporary_iterate = y_iterate;
		if (smoothness_known) step_size = 1.0 / smoothness;
		bool done = false;
		while (!done) {
                        retraction.retract(temporary_iterate, -1.0 * grad, step_size);
                        double F = evaluator.evaluate(temporary_iterate);
                        if (increase && !smoothness_known) {
                                if (F > Fval - rho * step_size * grad_norm_sq) increase = false;
                                else step_size /= eta;
                                temporary_iterate = y_iterate;
			} else {
//					printf("%1.30lf %3.30lf %3.30lf \n", step_size, (Fval - F) / (step_size * grad_norm_sq), sqrt(grad_norm_sq));
//					Print(y_iterate); printf("%lf \n \n", Fval);
//					Print(temporary_iterate); printf("%lf \n\n", F);
//					Print(grad);
				if (F <= Fval - rho * step_size * grad_norm_sq) {
					done = true;
				} else {
					temporary_iterate = y_iterate;
					step_size *= eta;
				}
			}
		}
		if (sqrt(grad_norm_sq) < gtol) {iterate = temporary_iterate; return it;}
                printf("%d %d %1.30lf %1.30lf \n", it, k, sqrt(grad_norm_sq), step_size);

                // Restart if there is not a sufficient decrease.
                if (evaluator.evaluate(temporary_iterate) > evaluator.evaluate(iterate) - restart_rho * step_size * grad_norm_sq) {
                        y_iterate = iterate;
			if (!smoothness_known) { // Vary step size to check change in restart period (if smoothness is not known).
				if (dec_step_size) {
					dec_step_size = false;
					if (k < period_tol * (prev_period / eta)) step_size /= eta;
				} else {
					step_size *= eta;
					dec_step_size = true;
					prev_period = k;
				}
			}
                        k = 0;
                } else {
                        // Apply momentum.
                        retraction.apply_momentum(y_iterate, iterate, temporary_iterate, k / (k + 3.0));
                        ++k;
                }
                ++it;
	}
}

}
}

#endif
