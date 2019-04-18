function X = CModesSmoothed1D

n = 1000;
k = 10;
mu = .1;
epsilon = 0.001; % Parameter determining how much the L1 is smoothed.

A = spdiags([-1.0*ones(n,1) 2.0*ones(n,1) -1.0*ones(n,1)], [-1,0,1], n , n) * n;

A(n,n) = A(n,n)/2;

% Generate initial point at random on the Stiefel manifold (testing robustness).
R = random('Normal', zeros(n,k), ones(n,k));
[X,~] = qr(R,0);

[X,it] = stiefel_opt_function_restart(X, @obj, @grad, 0, 1e-3, A, mu / n, epsilon);

for i=1:k
    figure; plot([0:1/n:1-1/n], X(:,i));
end

end

function [dF, PdFnorm, dFnorm] = grad(X, A, mu, epsilon)
    dF = A*X;
    dF = dF + mu*sign(X).*(abs(X) > epsilon) + (mu/epsilon)*X.*(abs(X) <= epsilon);
    dFnorm = norm(dF - (sqrt(2)/2)*X*(dF'*X) - (1 - sqrt(2)/2)*X*(X'*dF), 'fro');
    % Use A^{-1} as a preconditioner.
    dF_temp = A\dF;
    SM_1 = X'*dF_temp;
    SM_1 = (SM_1 + SM_1')/2;
    SM_2 = X'*(A\X);
    [V,D] = eig(SM_2);
    D = diag(D);
    SM_1 = V'*SM_1*V;
    S = -1.0 * SM_1 ./ (.5*(D*(ones(size(D))') + ones(size(D))*D'));
    dF_temp = dF_temp + (A\X)*(V*S*V');
    % Only use the preconditioner if it is a descent direction.
    if (trace(dF_temp'*dF) > 0)
        PdFnorm = sqrt(trace(dF_temp'*dF));
        dF = dF_temp - .5*X*(X'*dF_temp);
        fprintf("Tangency error in preconditioned grad: %f \n", norm(dF'*X + X'*dF, 'fro'));
    else
        PdFnorm = dFnorm;
        S = dF'*X;
        dF = (dF - 0.5 * X * (S + S'));
    end
end

function F = obj(X, A, mu, epsilon)
    F = .5*trace(X'*A*X) + sum(sum(mu*(abs(X) - epsilon/2).*(abs(X) > epsilon) ...
        + mu/(2*epsilon)*X.*X.*(abs(X) <= epsilon)));
end