function [ X, it ] = stiefel_opt_function_restart( X, f, gradf, L, tol, varargin )
%STIEFEL_OPT Optimizes a smooth function F on the Stiefel manifold.
% f must evaluate F and gradf must evaluate gradF.
% L is an estimate of the smoothness parameter of F.

r_rho = 0.01; % parameters controlling when we restart the method.
r_xi = 0.01;
Y = X;
it = 0;
n = 0;
step_size = 0.1;
alpha = 2;
G = feval(gradf, Y, varargin{:});
G = G - 0.5 * Y * (G' * Y + Y' * G);
normG = sqrt(norm(G, 'fro')^2 + norm(Y'*G, 'fro')^2);
F = feval(f, X, varargin{:});
while (normG > tol && step_size > 1e-13)
    X_temp = X;
    if L == 0
        X = retract(Y, -step_size*G);
        % Calculate step size using a line search.
        while feval(f, X, varargin{:}) < feval(f, Y, varargin{:}) - 0.7*step_size * normG^2
            step_size = step_size * alpha;
            X = retract(Y, -step_size*G);
        end
        while feval(f, X, varargin{:}) > feval(f, Y, varargin{:}) - 0.5*step_size * normG^2 && step_size > 1e-13
                % fprintf('F(X): %f F(Y): %f step_size: %f grad_norm: %f \n', feval(f, X, varargin{:}), feval(f, Y, varargin{:}), step_size, normG);
                step_size = step_size / alpha;
                X = retract(Y, -step_size*G);
        end
    else
        step_size = 1.0/L;
        X = retract(Y, -step_size*G);
    end
    Fnew = feval(f, X, varargin{:});
    if (Fnew > F - r_rho*step_size*normG^2)
        % fprintf('Restarted: %d \n', it);
        n = 0;
        X = X_temp;
        Y = X;
    else
        Y = apply_momentum(X_temp, X, n/(n+3.0));
        n = n + 1;
    end
    G = feval(gradf, Y, varargin{:});
    G = G - 0.5 * Y * (G' * Y + Y' * G);
    normG = sqrt(norm(G, 'fro')^2 + norm(Y'*G, 'fro')^2);
    F = feval(f, X, varargin{:});
    it = it + 1;
    % fprintf('Iteration Count: %d, Gradient Norm: %f, k: %d \n', it, normG, n);
end

end

