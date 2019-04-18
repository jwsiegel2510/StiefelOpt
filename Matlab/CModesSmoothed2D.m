function CModes = CModesSmoothed2D

n = 200;
k = 25;
mu = 5;
epsilon = .001;

w = ones(n) + 0*sin((1:n)*2*pi/n)' * sin((1:n)*2*pi/n);
pot = zeros(n);
pot(floor(0.35*n):floor(0.65*n), floor(0.35*n):floor(0.65*n)) = 0;

A = InhomoLaplace(w, pot/(n^2));

% Generate initial point at random on the Stiefel manifold (testing robustness).
R = random('Normal', zeros(n*n,k), ones(n*n,k));
[X,~] = qr(R,0);

[X,it] = stiefel_opt_function_restart(X, @obj, @grad, 0, 1e-4, A, mu/(n^2), epsilon);

CModes = [];
for i=1:k
    X(:,i) = sign(sum(X(:,i)))*X(:,i);
    eig = zeros(n);
    for j = 0:n-1
        for l = 0:n-1
            eig(j+1,l+1) = X(l + n*j + 1, i);
        end
    end
    CModes = [CModes eig];
end

[G,R] = meshgrid([0:1/n:1-1/n]);
SumCModes = zeros(n);
for i=0:k-1
    SumCModes = SumCModes + CModes(1:n, (i*n + 1):((i + 1)*n))
%    figure; mesh(G,R,CModes(1:n, (i*n + 1):((i + 1)*n)));
end
figure; imagesc(SumCModes);

end

function dF = grad(X, A, mu, epsilon)
    dF = A*X;
    dF = dF + mu*sign(X).*(abs(X) > epsilon) + (mu/epsilon)*X.*(abs(X) <= epsilon);
end

function F = obj(X, A, mu, epsilon)
    F = .5*trace(X'*A*X) + sum(sum(mu*(abs(X) - epsilon/2).*(abs(X) > epsilon) ...
        + mu/(2*epsilon)*X.*X.*(abs(X) <= epsilon)));
end