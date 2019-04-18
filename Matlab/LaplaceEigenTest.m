function Eigs = LaplaceEigenTest

n = 100;
k = 20;
w = ones(n) + 0.8*sin((1:n)*2*pi/n)' * sin((1:n)*2*pi/n);
A = InhomoLaplace(w, zeros(n));
X = zeros(n*n,k);
for i = 1:k
    X(i,i) = 1;
end

[X,~] = stiefel_opt_function_restart(X, @obj, @grad, 0, 1e-4, A);

Eigs = [];
for i=1:k
    eig = zeros(n);
    for j = 0:n-1
        for l = 0:n-1
            eig(j+1,l+1) = X(l + n*j + 1, i);
        end
    end
    Eigs = [Eigs eig];
end

for i=0:k-1
    figure; imagesc(Eigs(1:n, (i*n + 1):((i + 1)*n)));
end

end

function F = obj(X, A)
    s = size(X);
    F = 0.5*trace(X'*A*X*diag(1:1:s(2)));
end

function dF = grad(X, A)
    s = size(X);
    dF = A*X*diag(1:1:s(2));
end