function [ X, it ] = stiefel_opt_gradient_restart( X, f, gradf, tol, varargin )
% Optimizes a smooth function F on the Stiefel manifold.

Y = X;
it = 0;
n = 0;
G = feval(gradf, Y, varargin{:});
G = G - 0.5 * Y * (G' * Y + Y' * G);
normG = sqrt(norm(G, 'fro')^2 + norm(Y'*G, 'fro')^2);
step_size = 0.1;
alpha = 1.7;

while (normG > tol && step_size > 1e-13)
    X_temp = X;
    X = retract(Y, -step_size*G);
    % Calculate step size using a line search.
    while feval(f, X, varargin{:}) < feval(f, Y, varargin{:}) - 0.7*step_size * normG^2
        step_size = step_size * alpha;
        X = retract(Y, -step_size*G);
    end
    while feval(f, X, varargin{:}) > feval(f, Y, varargin{:}) - 0.5*step_size * normG^2 && step_size > 1e-13
            % fprintf('F(X): %f F(Y): %f step_size: %f \n', feval(f, X, varargin{:}), feval(f, Y, varargin{:}), step_size);
            step_size = step_size / alpha;
            X = retract(Y, -step_size*G);
            if step_size < 1e-8
                step_size = step_size;
            end
    end
    Y = apply_momentum(X_temp, X, n/(n+3.0));
    n = n + 1;
    G = feval(gradf, Y, varargin{:});
    G = G - 0.5 * Y * (G' * Y + Y' * G);
    normG = sqrt(norm(G, 'fro')^2 + norm(Y'*G, 'fro')^2);
    % Check restart condition.
    s = size(Y);
    Y_to_X = (2 * mldivide(eye(s(2)) + X'*Y, X'))';
    Y_to_X = Y_to_X - 0.5 * Y * (Y_to_X'*Y + Y'*Y_to_X);
    if trace(Y_to_X'*G + (Y_to_X'*Y)*(Y'*G)) < -1.0*step_size*normG^2
        % fprintf('Restarted: %d\n', it);
        n = 0;
        Y = X;
        G = feval(gradf, Y, varargin{:});
        G = G - 0.5 * Y * (G' * Y + Y' * G);
        normG = sqrt(norm(G, 'fro')^2 + norm(Y'*G, 'fro')^2);
    end
    it = it + 1;
    % fprintf('Iteration Count: %d, Gradient Norm: %f, k: %d, step_size: %f \n', it, normG, n, step_size);
end

end

