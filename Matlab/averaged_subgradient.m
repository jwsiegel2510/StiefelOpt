function [ X ] = averaged_subgradient( X, subgradf, step_size, iteration_count, varargin )
% Runs averaged subgradient on the Stiefel manifold with decreasing 
% step sizes as is optimal for strongly convex functions. step_size
% is the initial step size and iteration count is the total number
% of iterations.

Y = X;
for i = 1:iteration_count
    G = feval(subgradf, Y, varargin{:});
    Y = retract(Y, -step_size * G / sqrt(iteration_count));
    X = apply_momentum(X, Y, -(i-1)/(i+1));
end

end

